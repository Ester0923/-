from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ECA_Module(nn.Module):
    """轻量 ECA 模块，可插在 HGBlock 后或融合模块内."""

    def __init__(self, channel: int):
        super().__init__()
        k = int(abs(math.log2(channel)) + 1) | 1
        k = min(k, 13)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean(dim=(2, 3), keepdim=False).unsqueeze(1)  # (B,1,C)
        y = self.conv1d(y)  # 动态计算通道权重
        w = self.sigmoid(y).permute(0, 2, 1).unsqueeze(-1)  # (B,C,1,1)
        return x * w


class PriorHeatmapFusion_RTDETR(nn.Module):
    """优化版先验融合模块： 1. 修复训练中的参数注册与设备兼容问题 2. 提升性能，避免昂贵的张量操作 3. 限制调试输出，平衡调试与训练效率.
    """

    def __init__(
        self,
        in_channels: int,
        mode: str = "add",
        enable_dafpn: bool = True,
        use_spatial_attn: bool = True,
        prior_strength_base: float = 0.3,
        prior_strength_max: float = 0.8,
        add_k_std: float = 0.3,
        dropout_rate_prior: float = 0.03,
        dropout_rate_cat: float = 0.05,
        bound_scale: float = 0.3,
        residual_alpha: float = 0.1,
        learnable_alpha: bool = True,
        eps: float = 1e-6,
        debug: bool = False,
    ):
        super().__init__()
        assert mode in ("mul", "add", "cat")
        assert in_channels in (256, 512, 1024, 2048)
        self.in_channels = in_channels
        self.mode = mode
        self.enable_dafpn = enable_dafpn
        self.use_spatial_attn = use_spatial_attn
        self.prior_strength_base = prior_strength_base
        self.prior_strength_max = prior_strength_max
        self.add_k_std = add_k_std
        self.dropout_rate_prior = dropout_rate_prior
        self.dropout_rate_cat = dropout_rate_cat
        self.bound_scale = bound_scale
        self.eps = eps
        self.debug = debug

        # Prior正则化（训练用）
        self.prior_dropout = nn.Dropout2d(p=self.dropout_rate_prior) if self.dropout_rate_prior > 0 else None

        # 可学习残差系数
        self.learnable_alpha = learnable_alpha
        if self.learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(residual_alpha, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(residual_alpha, dtype=torch.float32))

        # DAFPN双重注意力（仅P3/P4层）
        self.channel_attn = None
        self.spatial_attn = None
        if self.enable_dafpn and self.in_channels in (512, 1024):
            k_ch = int(abs(math.log2(self.in_channels)) + 1) | 1
            k_ch = min(k_ch, 13)
            self.channel_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Unflatten(1, (1, self.in_channels)),
                nn.Conv1d(1, 1, kernel_size=k_ch, padding=k_ch // 2, bias=False),
                nn.Sigmoid(),
            )
            self.spatial_attn = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False), nn.BatchNorm2d(1), nn.Sigmoid()
            )

        # Cat模式专属模块
        self.eca_conv1d = None
        self.dw_conv = None
        self.pw_conv = None
        if self.mode == "cat":
            fused_channels = self.in_channels + 1
            # 补全ECA通道注意力
            k_eca = min(int(abs(math.log2(fused_channels)) + 1) | 1, 13)
            self.eca_conv1d = nn.Conv1d(1, 1, kernel_size=k_eca, padding=k_eca // 2, bias=False)
            self.eca_sigmoid = nn.Sigmoid()

            # 分组卷积
            g = math.gcd(fused_channels, in_channels)
            groups = max(1, g)
            self.conv = nn.Conv2d(fused_channels, in_channels, kernel_size=1, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(in_channels)
            self.cat_dropout = nn.Dropout2d(p=self.dropout_rate_cat) if self.dropout_rate_cat > 0 else None

            # 预创建深度可分离卷积（解决参数注册问题）
            self.dw_conv = nn.Conv2d(
                fused_channels, fused_channels, kernel_size=3, padding=1, groups=fused_channels, bias=False
            )
            self.pw_conv = nn.Conv2d(fused_channels, fused_channels, kernel_size=1, bias=False)
        else:
            self.eca_sigmoid = self.conv = self.bn = self.cat_dropout = None

        # Debug 打印控制
        self._debug_print_count = 0
        self._debug_print_limit = 200  # 限制最大打印次数

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.normal_(m.weight, mean=0.0, std=2e-3)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.9)
                nn.init.constant_(m.bias, 0.0)
        if self.eca_conv1d is not None:
            nn.init.constant_(self.eca_conv1d.weight, 0.0)
            self.eca_conv1d.weight.data[0, 0, self.eca_conv1d.kernel_size[0] // 2] = 0.2

    def _simam(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        mean = x_flat.mean(dim=2, keepdim=True)
        var = x_flat.var(dim=2, keepdim=True)
        energy = (x_flat - mean) ** 2 / (4.0 * (var + self.eps)) + 0.5
        attn = torch.sigmoid(energy).view(B, C, H, W)
        return attn.mean(dim=1, keepdim=True)

    def _deformable_sampling(self, feat: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        ds = 8 if H >= 80 else 4 if H >= 40 else 2
        small_prior = F.adaptive_avg_pool2d(prior, (max(1, H // ds), max(1, W // ds)))
        out_feat = feat.clone()

        for b in range(B):
            flat_prior = small_prior[b].view(-1)
            if flat_prior.numel() == 0 or flat_prior.max() < 0.2:
                continue
            _vals, idx = torch.topk(flat_prior, min(10, flat_prior.numel()))
            coords = torch.stack([idx // small_prior.shape[3], idx % small_prior.shape[3]], dim=1)

            samples = []
            for y, x in coords:
                yy = int((y.item() + 0.5) * ds)
                xx = int((x.item() + 0.5) * ds)
                yy = min(max(0, yy), H - 1)
                xx = min(max(0, xx), W - 1)
                patch = feat[b : b + 1, :, max(0, yy - 1) : min(H, yy + 2), max(0, xx - 1) : min(W, xx + 2)]
                if patch.numel() == 0:
                    continue
                samples.append(F.adaptive_avg_pool2d(patch, (1, 1)).squeeze(-1).squeeze(-1))

            if len(samples) > 0:
                sampled_feat = torch.stack(samples, dim=0).mean(dim=0)
                out_feat[b] = out_feat[b] * 0.6 + sampled_feat.view(C, 1, 1) * 0.4
        return out_feat

    def _sanitize_prior(
        self,
        prior: torch.Tensor | None,
        feat_shape: tuple[int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
        is_confidence: bool = False,
    ) -> torch.Tensor:
        B, _, H, W = feat_shape
        if prior is None:
            fill = 1.0 if is_confidence else 0.0
            return torch.full((B, 1, H, W), fill, device=device, dtype=dtype)

        if not isinstance(prior, torch.Tensor):
            prior = torch.as_tensor(prior, device=device, dtype=dtype)
        if prior.dim() == 0:
            prior = prior.view(1, 1, 1, 1).repeat(B, 1, H, W)
        elif prior.dim() == 2:
            prior = prior.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
        elif prior.dim() == 3:
            prior = prior.unsqueeze(1)
        assert prior.dim() == 4, f"先验维度需为0/2/3/4，当前: {prior.dim()}"

        if prior.shape[0] != B:
            prior = prior.repeat((B + prior.shape[0] - 1) // prior.shape[0], 1, 1, 1)[:B]
        if prior.shape[1] != 1:
            prior = prior[:, :1]

        # 明确插值参数（修复潜在的版本兼容问题）
        if (prior.shape[2], prior.shape[3]) != (H, W):
            interp_mode = "bilinear" if H >= 40 else "nearest"
            if interp_mode == "bilinear":
                prior = F.interpolate(prior, size=(H, W), mode=interp_mode, align_corners=False)
            else:
                prior = F.interpolate(prior, size=(H, W), mode=interp_mode)

        return torch.clamp(prior, min=self.eps * 1e-1, max=1.0 - self.eps * 1e-1)

    def forward(
        self,
        feat: torch.Tensor,
        prior: torch.Tensor | None = None,
        prior_confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert isinstance(feat, torch.Tensor) and feat.dim() == 4
        B, C, H, W = feat.shape
        assert C == self.in_channels, f"通道不匹配：配置{self.in_channels} vs 输入{C}"

        device, dtype = feat.device, feat.dtype
        prior = self._sanitize_prior(prior, (B, C, H, W), device, dtype, is_confidence=False)
        prior_confidence = self._sanitize_prior(prior_confidence, (B, C, H, W), device, dtype, is_confidence=True)
        if self.training and self.prior_dropout is not None:
            prior = self.prior_dropout(prior)
        weighted_prior = prior * prior_confidence

        # 高效的透明化判断（替换allclose，提升性能）
        if weighted_prior.abs().max().item() < 1e-6:
            if self.debug and self.training and self._debug_print_count < self._debug_print_limit:
                print(f"[Fusion Debug] {H}x{W} | weighted_prior all-zero -> transparent")
                self._debug_print_count += 1
            return feat

        # 受限的调试输出（避免拖慢训练）
        if self.debug and self.training and self._debug_print_count < self._debug_print_limit:
            wp_mean = weighted_prior.mean().item()
            wp_max = weighted_prior.max().item()
            print(f"[Fusion Debug] {H}x{W} | prior mean: {wp_mean:.6f}, max: {wp_max:.6f}")
            self._debug_print_count += 1

        # DAFPN双重注意力
        if self.enable_dafpn and self.channel_attn is not None and self.spatial_attn is not None:
            ch_w = self.channel_attn(feat).permute(0, 2, 1).unsqueeze(-1)
            feat = feat * ch_w
            sp_w = self.spatial_attn(feat.mean(dim=1, keepdim=True))
            feat = feat * sp_w

        # 先验区域特征强化
        if self.mode == "add" and weighted_prior.max().item() > 0.1:
            prior_mask = (weighted_prior > 0.2).float()
            feat = feat * (1.0 + 0.5 * prior_mask)

        # 融合模式执行
        if self.mode == "mul":
            conf_mean = prior_confidence.mean(dim=[1, 2, 3], keepdim=True)
            dyn_strength = (
                self.prior_strength_base + (self.prior_strength_max - self.prior_strength_base) * conf_mean
            ).clamp(min=self.prior_strength_base, max=self.prior_strength_max)
            gain = 1.0 + dyn_strength * weighted_prior.clamp(max=0.6)
            out = feat * gain

        elif self.mode == "add":
            grid_size = 8 if H == 80 else 4
            local_mean = F.adaptive_avg_pool2d(feat.abs(), (grid_size, grid_size))
            local_sq_mean = F.adaptive_avg_pool2d(feat.abs() ** 2, (grid_size, grid_size))
            local_std = (local_sq_mean - local_mean**2).clamp(min=0.0).sqrt()

            interp_mode = "bilinear" if H >= 40 else "nearest"
            local_mean_up = F.interpolate(
                local_mean, size=(H, W), mode=interp_mode, align_corners=False if interp_mode == "bilinear" else None
            )
            local_std_up = F.interpolate(
                local_std, size=(H, W), mode=interp_mode, align_corners=False if interp_mode == "bilinear" else None
            )

            bound = (local_mean_up + self.add_k_std * local_std_up).mean(dim=1, keepdim=True) * self.bound_scale
            add_term = self.prior_strength_base * weighted_prior
            add_term = torch.minimum(add_term, bound.clamp(max=feat.abs().mean() * 0.5))
            out = feat + add_term

        elif self.mode == "cat":
            fused = torch.cat([feat, weighted_prior], dim=1)
            # 使用预创建的dw_conv/pw_conv（解决参数未被优化问题）
            if self.in_channels >= 1024:
                fused = self._deformable_sampling(fused, weighted_prior)
                if self.dw_conv is not None and self.pw_conv is not None:
                    fused = self.dw_conv(fused)
                    fused = self.pw_conv(fused)

            g = fused.mean(dim=(2, 3)).unsqueeze(1)
            att = self.eca_conv1d(g)
            att = self.eca_sigmoid(att).permute(0, 2, 1).unsqueeze(-1)
            fused = fused * att

            if self.training and self.use_spatial_attn and H >= 40:
                fused = fused * self._simam(fused)

            fused = self.conv(fused)
            fused = self.bn(fused)
            fused = F.relu(fused, inplace=False)
            if self.training and self.cat_dropout is not None:
                fused = self.cat_dropout(fused)
            out = fused

        else:
            raise RuntimeError(f"不支持的模式：{self.mode}")

        # 残差融合
        alpha = torch.clamp(self.alpha, 0.0, 0.5)
        return feat + alpha * (out - feat)


if __name__ == "__main__":

    def _verify():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"验证设备：{device}")
        B = 2

        # 验证P3层（512通道，80x80）
        p3_feat = torch.randn(B, 512, 80, 80, device=device)
        p3_fusion = PriorHeatmapFusion_RTDETR(in_channels=512, mode="add", enable_dafpn=True, debug=True).to(device)
        prior_crazing = torch.zeros(B, 80, 80, device=device)
        prior_crazing[:, 20:60, 20:60] = 0.6
        out_p3 = p3_fusion(p3_feat, prior=prior_crazing, prior_confidence=torch.ones(B, 1, 80, 80, device=device))
        print(f"P3层先验-特征差异：{(out_p3 - p3_feat).abs().mean().item():.6f}")

        # 验证P5层（2048通道，20x20）
        p5_feat = torch.randn(B, 2048, 20, 20, device=device)
        p5_fusion = PriorHeatmapFusion_RTDETR(
            in_channels=2048, mode="add", enable_dafpn=False, prior_strength_base=0.1
        ).to(device)
        out_p5 = p5_fusion(p5_feat, prior=None)
        print(f"P5层无先验差异：{(out_p5 - p5_feat).abs().mean().item():.6f}")

    _verify()
