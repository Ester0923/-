import os
from pathlib import Path

def generate_dataset_txt():
    # ================= 配置区域 =================
    # 你的数据集根目录
    base_dir = '/mnt/bigdisk/ultralytics/data/severstal'
    
    # 需要处理的子集名称
    subsets = ['train', 'test','valid']
    
    # 支持的图片后缀 (NEU-DET 通常是 .jpg 或 .bmp，这里包含常见格式)
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    # ===========================================

    for subset in subsets:
        # 定义路径
        label_dir = os.path.join(base_dir, subset, 'labels')
        # 假设图片存放在与 labels 平级的 images 文件夹中（YOLO标准结构）
        image_dir = os.path.join(base_dir, subset, 'images') 
        output_txt_path = os.path.join(base_dir, f'{subset}.txt')

        # 检查文件夹是否存在
        if not os.path.exists(label_dir):
            print(f"[错误] 找不到标签文件夹: {label_dir}")
            continue
        
        if not os.path.exists(image_dir):
            print(f"[警告] 找不到图片文件夹: {image_dir} (脚本将尝试在labels同级查找，但建议检查结构)")
            # 如果 images 文件夹不存在，可能图片和标签混在一起，或者路径不对
            # 这里暂时保持 image_dir 不变，后续判断会提示找不到图片

        print(f"正在处理: {subset} ...")
        
        valid_paths = []
        
        # 获取所有 txt 标签文件
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        label_files.sort() # 排序，保证顺序一致

        missing_count = 0
        
        for label_file in label_files:
            # 获取文件名（不含后缀），例如 'crazing_1'
            file_stem = Path(label_file).stem
            
            # 寻找对应的图片文件
            found_img = False
            for ext in valid_exts:
                img_name = file_stem + ext
                img_path = os.path.join(image_dir, img_name)
                
                if os.path.exists(img_path):
                    valid_paths.append(img_path)
                    found_img = True
                    break
            
            if not found_img:
                missing_count += 1
                # 如果你想看具体的报错，可以取消下面这行的注释
                # print(f"  未找到对应图片: {file_stem} (在 {image_dir})")

        # 写入 txt 文件
        with open(output_txt_path, 'w') as f:
            for path in valid_paths:
                f.write(path + '\n')

        print(f"--> 完成！已生成: {output_txt_path}")
        print(f"    共写入路径: {len(valid_paths)} 条")
        if missing_count > 0:
            print(f"    [注意] 有 {missing_count} 个标签没找到对应的图片，已跳过。")
        print("-" * 30)

if __name__ == '__main__':
    generate_dataset_txt()