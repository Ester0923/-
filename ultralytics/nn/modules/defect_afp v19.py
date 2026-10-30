import torch
import torch.nn.functional as F
from torch import nn


# =========================================================================
# 🏗️ 0. 基础组件
# =========================================================================
class StaticSobel(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        # 固定权重的 Sobel 算子 (不可学习) -> 纯粹的物理先验
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("wx", sobel_x.view(1, 1, 3, 3).repeat(in_ch, 1, 1, 1))
        self.register_buffer("wy", sobel_y.view(1, 1, 3, 3).repeat(in_ch, 1, 1, 1))
        self.groups = in_ch

    def forward(self, x):
        gx = F.conv2d(x, self.wx, padding=1, groups=self.groups)
        gy = F.conv2d(x, self.wy, padding=1, groups=self.groups)
        # 返回梯度模长
        return torch.sqrt(gx**2 + gy**2 + 1e-6)


class DWConv(nn.Module):
    """深度可分离卷积 (轻量化)."""

    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.c(x)


class FPFA_Block(nn.Module):
    """特征金字塔融合适配块."""

    def __init__(self, sem_ch, det_ch, out_ch):
        super().__init__()
        self.proj_sem = nn.Conv2d(sem_ch, out_ch, 1)
        self.proj_det = nn.Conv2d(det_ch, out_ch, 1)
        self.up = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1, bias=False)
        self.fuse = DWConv(out_ch, out_ch)

    def forward(self, f_sem, f_det):
        return self.fuse(self.proj_det(f_det) + self.up(self.proj_sem(f_sem)))


# =========================================================================
# 🔥 1. 轻量级梯度校准器 (LightGradientCalibrator)
#    逻辑：Sobel -> 1x1 Conv -> BN -> SiLU
# =========================================================================
class LightGradientCalibrator(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.sobel = StaticSobel(ch)
        # 简单的特征映射，让 Sobel 特征能融入语义空间
        self.calibrator = nn.Sequential(nn.Conv2d(ch, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.SiLU())

    def forward(self, x):
        raw_edge = self.sobel(x)
        return self.calibrator(raw_edge)


# =========================================================================
# 🔥 2. 暴力混合块 (AggressiveHybridBlock / DIRB)
#    核心逻辑：Semantic + beta * Gradient
#    这就是 v19 成功的关键：直接注入，拒绝门控饱和！
# =========================================================================
class AggressiveHybridBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_gradient=True):
        super().__init__()
        self.use_gradient = use_gradient
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)

        # 语义分支 (保留 CNN 提取能力)
        self.semantic = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU()
        )

        if self.use_gradient:
            # 梯度分支
            self.reduce = nn.Conv2d(out_ch, out_ch // 2, 1)
            self.grad_calib = LightGradientCalibrator(out_ch // 2)
            self.edge_proj = nn.Conv2d(out_ch // 2, out_ch, 1, bias=False)

            # 🔥 强制注入系数：初始化为 0.5，强迫网络一开始就看梯度
            self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        self.fuse = nn.Sequential(nn.Conv2d(out_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU())

    def forward(self, x):
        x = self.proj(x)
        feat_sem = self.semantic(x)

        edge_out = None  # 用于 Loss 监督 (Residuals)

        if self.use_gradient:
            # 1. 提取梯度
            edge_small = self.grad_calib(self.reduce(x))
            # 2. 映射回原通道
            edge_gate = self.edge_proj(edge_small)

            # 保存残差
            edge_out = edge_gate

            # 3. 🔥 强制加法注入 (Direct Injection) 🔥
            # 这种“硬连接”保证了梯度不会消失
            feat_enhanced = feat_sem + self.beta * edge_gate
        else:
            feat_enhanced = feat_sem
            # 如果不开启梯度，残差为 0
            if self.training:
                edge_out = torch.zeros_like(feat_sem)

        return self.fuse(feat_enhanced + x), edge_out


# =========================================================================
# 🚀 3. 主网络: DyReconstructNet v19 (Final)
# =========================================================================
class DyReconstructNet(nn.Module):
    def __init__(self, in_ch_list, out_ch=256):
        super().__init__()

        # --- 修复参数解析 Bug (防止 list/int 类型错误) ---
        if isinstance(out_ch, (list, tuple)):
            out_ch = out_ch[0]
        new_in_ch = []
        for ch in in_ch_list:
            if isinstance(ch, (list, tuple)):
                new_in_ch.append(ch[0])
            else:
                new_in_ch.append(ch)
        in_ch_list = new_in_ch

        # 必须定义 out_channels 供 tasks.py 使用
        self.out_channels = [out_ch, out_ch, out_ch]

        self.sem_p5_adapter = nn.Conv2d(in_ch_list[2], out_ch, 1)
        self.fpfa_p4 = FPFA_Block(out_ch, in_ch_list[1], out_ch)
        self.fpfa_p3 = FPFA_Block(out_ch, in_ch_list[0], out_ch)

        # 🔥 配置策略：底层强注入，高层纯语义 🔥
        # P3 (80x80): 纹理最丰富 -> 开启梯度
        self.hpm_p3 = AggressiveHybridBlock(in_ch_list[0], out_ch, use_gradient=True)
        # P4 (40x40): 仍有纹理 -> 开启梯度
        self.hpm_p4 = AggressiveHybridBlock(in_ch_list[1], out_ch, use_gradient=True)
        # P5 (20x20): 纯语义 -> 关闭梯度 (避免引入高层噪声)
        self.hpm_p5 = AggressiveHybridBlock(in_ch_list[2], out_ch, use_gradient=False)

        self.fuse_p5 = DWConv(out_ch, out_ch)
        self.fuse_p4 = DWConv(out_ch, out_ch)
        self.fuse_p3 = DWConv(out_ch, out_ch)

    def forward(self, feats):
        p3, p4, p5 = feats

        # 1. 语义流 (Top-down)
        s5 = self.sem_p5_adapter(p5)
        s4 = self.fpfa_p4(s5, p4)
        s3 = self.fpfa_p3(s4, p3)

        # 2. 纹理重构 (In-place Restoration)
        d5, res5 = self.hpm_p5(p5)
        d4, res4 = self.hpm_p4(p4)
        d3, res3 = self.hpm_p3(p3)

        # 3. 最终融合
        o5 = self.fuse_p5(s5 + d5)
        o4 = self.fuse_p4(s4 + d4)
        o3 = self.fuse_p3(s3 + d3)

        outputs = [o3, o4, o5]
        # P5 的 res5 是 0，loss.py 会自动忽略或计算为0
        residuals = [res3, res4, res5]

        if self.training:
            return outputs, residuals
        return outputs
