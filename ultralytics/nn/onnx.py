import torch
from ultralytics.nn.modules.defect_afp import DefectAFP_Lite  # 按你的路径导入

# 将三路输入打包，便于 trace/导出
class Wrapper(torch.nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core
    def forward(self, p3, p4, p5):
        outs = self.core([p3, p4, p5])   # 你的 forward 期望 list[Tensor]
        return tuple(outs)               # 让可视化里有 3 个明确输出端

def build_model():
    return DefectAFP_Lite(
        in_ch_list=[512, 1024, 2048],
        out_ch=256,
        se_reduction=None,
        fixed_kernel_ratio=0.6,
        trainable_fixed_kernel=False,
        num_directions=2,
        enable_bifpn=True,
        enable_afp=True,
        bifpn_type="full",
        use_adaptive_fusion=True
    )

if __name__ == "__main__":
    model = build_model().eval()

    # 如需带权重可视化，先加载：
    # sd = torch.load("checkpoints/defectafp_lite_weights.pth", map_location="cpu")
    # model.load_state_dict(sd, strict=True)

    wrap = Wrapper(model).eval()

    # 示例输入（与实际特征分辨率一致更直观）
    ex_p3 = torch.randn(1, 512, 100, 100)
    ex_p4 = torch.randn(1, 1024, 50, 50)
    ex_p5 = torch.randn(1, 2048, 25, 25)

    with torch.no_grad():
        traced = torch.jit.trace(wrap, (ex_p3, ex_p4, ex_p5), strict=False)
        traced.save("defectafp_lite_ts.pt")
        print("TorchScript 已导出: defectafp_lite_ts.pt")

    # 若更偏好 ONNX，可再导出一份：
    torch.onnx.export(
        wrap, (ex_p3, ex_p4, ex_p5), "defectafp_lite.onnx",
        input_names=["p3", "p4", "p5"],
        output_names=["out_p3", "out_p4", "out_p5"],
        opset_version=13, do_constant_folding=True,
        dynamic_axes={n: {0: "batch"} for n in ["p3","p4","p5","out_p3","out_p4","out_p5"]}
    )
    print("ONNX 已导出: defectafp_lite.onnx")
