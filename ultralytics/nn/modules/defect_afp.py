import torch
import torch.nn.functional as F
from torch import nn


# ==========================================
# 1. 基础轻量化组件 (保留自原文件)
# ==========================================
class DWConv(nn.Module):
    """Depthwise Separable Convolution for Feature Reconstruction."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class SEBlock(nn.Module):
    """Global Squeeze-and-Excitation operator applied to U_fused."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ==========================================
# 2. AFEM 四大核心特征抽取分支 (保留自原文件)
# ==========================================
class FrequencyBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
        self.register_buffer("laplacian_kernel", kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1))
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pwconv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        high_freq = F.conv2d(x, self.laplacian_kernel, stride=1, padding=1, groups=x.shape[1])
        out = self.dwconv(high_freq)
        out = self.pwconv(out)
        return self.act(self.bn(out))


class DirectionBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        kernels = [
            [[0, 0, 0], [-1, 2, -1], [0, 0, 0]],
            [[0, -1, 0], [0, 2, 0], [0, -1, 0]],
            [[-1, 0, 0], [0, 2, 0], [0, 0, -1]],
            [[0, 0, -1], [0, 2, 0], [-1, 0, 0]],
        ]
        self.dwconvs = nn.ModuleList()
        for k in kernels:
            dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
            dw.weight.data = torch.tensor(k, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
            self.dwconvs.append(dw)

        self.fused_conv = nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        dir_feats = [dw(x) for dw in self.dwconvs]
        out = self.fused_conv(torch.cat(dir_feats, dim=1))
        return self.act(self.bn(out))


class AFEM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.f_s = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.f_c = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels))
        self.f_f = FrequencyBranch(channels)
        self.f_d = DirectionBranch(channels)

        self.g_f = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels))
        self.g_d = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels))

        self.gamma_sf = nn.Parameter(torch.tensor(1.0))
        self.gamma_cd = nn.Parameter(torch.tensor(1.0))

        self.se = SEBlock(channels * 6)
        self.refine = nn.Sequential(
            DWConv(channels * 6, channels),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        U_s, U_c, U_f, U_d = self.f_s(x), self.f_c(x), self.f_f(x), self.f_d(x)
        U_sf = U_s * (self.gamma_sf * torch.sigmoid(self.g_f(U_f)))
        U_cd = U_c * (self.gamma_cd * torch.sigmoid(self.g_d(U_d)))
        U_fused = torch.cat([U_s, U_c, U_f, U_d, U_sf, U_cd], dim=1)
        U_refined = self.refine(self.se(U_fused))
        return U_refined + x


class AFM(nn.Module):
    def __init__(self, channels, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, A_s, B_s):
        """A_s: 语义流(BiFPN), B_s: 细节流(AFEM)."""
        A_norm = F.normalize(A_s, p=2, dim=1)
        B_norm = F.normalize(B_s, p=2, dim=1)
        beta_s = torch.clamp(self.beta, 0.0, 1.0)
        Z_s = (1.0 - beta_s) * A_norm + beta_s * B_norm + self.alpha * A_s
        return Z_s


# ==========================================
# 3. 从两张图片中提取并还原的 FullBiFPN
# ==========================================
class FullBiFPN(nn.Module):
    def __init__(self, in_ch_list, out_ch):
        super().__init__()
        self.num_scales = len(in_ch_list)
        self.epsilon = 1e-4

        # 通道对齐
        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_ch, out_ch, kernel_size=1) for in_ch in in_ch_list])

        # 每个融合节点使用2个输入，定义可学习权重
        self.top_down_weights = nn.ParameterList(
            [nn.Parameter(torch.ones(2, dtype=torch.float32)) for _ in range(self.num_scales - 1)]
        )
        self.bottom_up_weights = nn.ParameterList(
            [nn.Parameter(torch.ones(2, dtype=torch.float32)) for _ in range(self.num_scales - 1)]
        )

        # 融合卷积 (带 BN 和 激活函数，与图纸完美对应)
        self.top_down_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True)
                )
                for _ in range(self.num_scales - 1)
            ]
        )
        self.bottom_up_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True)
                )
                for _ in range(self.num_scales - 1)
            ]
        )

    def forward(self, feats):
        # 输入: [P3, P4, P5]
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, feats)]

        # 1. 自顶向下融合
        top_down = [laterals[-1]]  # 从 P5 开始
        for i in range(self.num_scales - 2, -1, -1):
            # 仅使用两个输入：上一层特征（上采样）+ 当前层特征
            inputs = [self.upsample(top_down[0]), laterals[i]]

            # 加权融合
            w = torch.relu(self.top_down_weights[self.num_scales - 2 - i])
            weighted_sum = (w[0] * inputs[0] + w[1] * inputs[1]) / (w.sum() + self.epsilon)
            # 插入到列表头部，方便下一轮上采样
            top_down.insert(0, self.top_down_convs[self.num_scales - 2 - i](weighted_sum))

        # 2. 自底向上融合
        bottom_up = [top_down[0]]  # 从 P3 开始
        for i in range(1, self.num_scales):
            # 仅使用两个输入：下一层特征（下采样）+ 当前层特征
            inputs = [self.downsample(bottom_up[-1]), top_down[i]]

            # 加权融合
            w = torch.relu(self.bottom_up_weights[i - 1])
            weighted_sum = (w[0] * inputs[0] + w[1] * inputs[1]) / (w.sum() + self.epsilon)
            bottom_up.append(self.bottom_up_convs[i - 1](weighted_sum))

        return bottom_up  # 输出[P3_out, P4_out, P5_out]


# ==========================================
# 4. 顶层整合模块: DefectAFP (双流网络完整版)
# ==========================================
class DefectAFP(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        # 1. 语义流 Semantic Path: 初始化 FullBiFPN
        self.bifpn = FullBiFPN(in_channels_list, out_channels)

        self.align_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True)
                )
                for c in in_channels_list
            ]
        )

        # 为每个尺度独立实例化细节增强模块（AFEM）与自适应流融合锁（AFM）
        self.afem_paths = nn.ModuleList([AFEM(out_channels) for _ in in_channels_list])
        self.afm_locks = nn.ModuleList([AFM(out_channels) for _ in in_channels_list])

    def forward(self, feats):
        # 兜底防御，确保多尺度特征传入格式正常
        if not isinstance(feats, (list, tuple)):
            feats = [feats]

        # ========= 步骤1：获取语义流特征 (FullBiFPN) =========
        # A_s_list 包含 [P3', P4', P5']，通道数均为 out_channels
        A_s_list = self.bifpn(feats)

        out_list = []
        for i, f in enumerate(feats):
            # ========= 步骤2：获取细节流特征 (通道对齐 + AFEM) =========
            x = self.align_convs[i](f)
            B_s = self.afem_paths[i](x)  # B_s: 细节流平行增强特征

            # ========= 步骤3：跨流自适应融合 =========
            A_s = A_s_list[i]  # A_s: 来自 FullBiFPN 的语义特征

            # 通过 AFM 自适应凸组合融合 A_s (语义) 和 B_s (细节)
            Z_s = self.afm_locks[i](A_s, B_s)
            out_list.append(Z_s)

        # 返回最终融合好之后的特征列表，可以直接接YOLO/Faster-RCNN的Head
        return out_list


# ==========================================
# 测试 / 使用示例
# ==========================================
if __name__ == "__main__":
    # 模拟骨干网络输出的 P3, P4, P5
    feats = [
        torch.randn(2, 256, 80, 80),  # P3
        torch.randn(2, 512, 40, 40),  # P4
        torch.randn(2, 1024, 20, 20),  # P5
    ]

    # 实例化结合后的颈部网络(Neck)
    neck = DefectAFP(in_channels_list=[256, 512, 1024], out_channels=256)

    # 前向传播
    outputs = neck(feats)

    for i, out in enumerate(outputs):
        print(f"Output P{i + 3} shape: {out.shape}")
