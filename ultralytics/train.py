import warnings

from ultralytics import RTDETR

warnings.filterwarnings("ignore")
# export PYTHONPATH=/mnt/bigdisk/ultralytics:$PYTHONPATH
model = RTDETR(r"/mnt/bigdisk/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml")
model.load("/mnt/bigdisk/ultralytics/runs/detect/train95/weights/best.pt")  # 是否加载预训练权重
# /mnt/bigdisk/ultralytics/runs/detect/train26/weights/best.pt

model.train(
    data="/mnt/bigdisk/ultralytics/ultralytics/data.yaml",
    imgsz=800,
    batch=12,
    epochs=300,
    rect=True,  # 🔴 优化：开启矩形训练。你的槟榔图像长宽比极大，开启后能有效减少由于强行填充正方形引入的白边白噪。
    mosaic=0.5,  # 🔴 降权：槟榔图像背景极度纯净，Mosaic 太高会造成目标大面积跨边界拼接，破坏植物长条形主干的空间连续性。
    mixup=0.1,  # 🔴 激活：背光半透明边缘会产生边缘交织。给 0.1 弱混叠，可以增强自适应融合模块（AFM）在弱对比度环境下的判别力。
    copy_paste=0.3,  # 🔴 激活：非常建议开启 0.3！将一个槟榔实例随机复制到另一个纯白背景图上，能在二值场景下瞬间扩充样本规模，且不会引入假边缘。
    degrees=15.0,  # 🔥 强化：调高旋转角度！槟榔在皮带或流道上是不规则任意角度倾斜的。你的 DirectionBranch 拥有 0°/90°/±45° 各向异性卷积核，加大旋转能彻底激发模型对细长划痕和开裂的旋转不变性识别。
    scale=0.2,
    shear=2.0,  # 🔥 强化：调高剪切形变。模拟槟榔在相机边缘由于广角镜头畸变产生的物理拉伸。
    flipud=0.2,
    fliplr=0.2,
    erasing=0,  # 📉 调低：背光图像中目标面积占比本就不大，擦除率 0.4 太过暴力，极易把端点处的小缺陷整个抹去，降到 0.2 更安全。
    optimizer="AdamW",
    lr0=0.0001,
    lrf=0.005,
    cos_lr=True,
    weight_decay=0.0001,  # 📈 调高：对于纯白高对比度背景，微调极易过拟合。从 0.0001 调高到 0.0005，强行约束模型只去学习核心缺陷特征，不背诵特定背景。
    warmup_epochs=15,  # 📈 调高：给模型 15 个 Epoch 的平缓预热，让细节增强模块（AFEM）里的 4 个可学习门控标量和 AFM 的平衡因子（beta_s）有一个平稳梯度初始化的周期。
    close_mosaic=40,  # 📅 提前关闭：在最后 40 个 Epoch 彻底关闭 Mosaic。让网络回归到最纯净的槟榔实拍图上，进行边界框（Bounding Box）高严格度 IoU 拟合。
    patience=50,
    workers=4,
    device=0,
    amp=True,
)
