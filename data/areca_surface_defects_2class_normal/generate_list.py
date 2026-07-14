import os
from pathlib import Path

# 1. 定义数据集根目录
dataset_root = "/mnt/bigdisk/ultralytics/data/areca_surface_defects_2class_normal"

# 2. 定义需要处理的三个子集 (名称, 相对路径标签夹, 最终输出文件名)
tasks = [
    ("训练集", "train", "train.txt"),
    ("验证集", "val", "val.txt"),
    ("测试集", "test", "test.txt")
]

# 支持的图片后缀名列表
img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG']

print("==================== 开始处理数据集列表 ====================")

for label_name, folder_name, output_file in tasks:
    # 动态拼接每一组的标签目录和输出文件绝对路径
    labels_dir = os.path.join(dataset_root, folder_name, "labels")
    output_txt = os.path.join(dataset_root, output_file)
    
    # 智能检查：如果该子集的标签文件夹不存在，则自动跳过
    if not os.path.exists(labels_dir):
        print(f"ℹ️  提示：未检测到{label_name}目录 ({labels_dir})，已自动跳过。")
        continue
        
    image_paths = []
    # 遍历当前子集的所有 txt 标签
    for file in os.listdir(labels_dir):
        if file.endswith('.txt'):
            base_name = Path(file).stem
            
            # 去配对的图片目录寻找对应的图片
            possible_img_dirs = [
                os.path.join(dataset_root, folder_name, "images"),
                os.path.join(dataset_root, "images", folder_name),
                os.path.join(dataset_root, folder_name)
            ]
            
            img_found = False
            for img_dir in possible_img_dirs:
                for ext in img_extensions:
                    potential_img_path = os.path.join(img_dir, base_name + ext)
                    if os.path.exists(potential_img_path):
                        image_paths.append(potential_img_path)
                        img_found = True
                        break
                if img_found:
                    break
                    
            if not img_found:
                print(f"⚠️  警告：[{label_name}] 标签 {file} 找不到对应的图片文件！")

    # 将图片绝对路径写入对应的总文本中
    with open(output_txt, 'w', encoding='utf-8') as f:
        for path in sorted(image_paths):
            f.write(path + '\n')

    print(f"✅ 成功！已生成{label_name}总列表：{output_file} (共 {len(image_paths)} 张图片)")

print("============================================================")