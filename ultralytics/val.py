import warnings

from ultralytics import RTDETR

warnings.filterwarnings("ignore")
# export PYTHONPATH=/mnt/bigdisk/ultralytics:$PYTHONPATH

# 1. 加载你训练好的最佳权重
# 请把下面的路径换成你训练出来的那个 best.pt
model_path = "/mnt/bigdisk/ultralytics/runs/detect/train48/weights/best.pt"
model = RTDETR(model_path)

# 2. 运行验证模式 (开启 TTA)
print("正在进行 TTA 验证，请稍候...")
metrics = model.val(
    data="/mnt/bigdisk/ultralytics/ultralytics/data.yaml",
    # 🔥 关键参数 🔥
    imgsz=800,  # 必须和你训练时保持一致！
    rect=True,  # 必须和你训练时保持一致！(去黑边)
    augment=True,  # ✅✅✅ 这里才是开启 TTA 的地方！
    # 它会自动把图片翻转、缩放预测多次，然后合并结果。
    batch=1,  # 开启 TTA 时建议 batch=1，防止不同尺度显存溢出
    device=0,
)

# 3. 打印结果
print(f"TTA mAP50: {metrics.box.map50}")
print(f"TTA mAP50-95: {metrics.box.map}")
