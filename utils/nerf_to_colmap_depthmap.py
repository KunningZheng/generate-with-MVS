import json
import numpy as np
import os
import shutil

def clean_filename(filepath):
    """
    清理文件路径，生成一个唯一的、扁平化的文件名。
    例如：../../train/block_1/0001.png -> block_1_0001.png
    """
    parts = os.path.normpath(filepath).split(os.sep)
    # 过滤掉 ".." 和 "train" (如果 "train" 总是根目录)
    cleaned_parts = [p for p in parts if p not in ('..', 'train') and p != '']
    # 将父文件夹和文件名连接起来（例如 block_1_0001.png）
    return "_".join(cleaned_parts)




def convert_nerf_to_colmap(data, sparse_path):

    # ==== 3. Write images.txt ====
    # 记录文件名对应关系
    filename_mapping = {} 
    for frame in data["frames"]:
        original_filepath = frame["file_path"]
        new_filename = clean_filename(original_filepath).replace('.png', '.exr')

        depth_folder = original_filepath.replace("../../train/", "").split('/')[0]+"_depth"
        original_depthpath = os.path.join(depth_folder, os.path.basename(original_filepath).replace('.png', '.exr'))
        filename_mapping[new_filename] = original_depthpath
        


    return filename_mapping



image_original_path = "/media/rylynn/data/MatrixCity/"
current_path = "/home/rylynn/Pictures/datasets_3Dline/MatrixCity/block_B"
image_path = os.path.join(current_path, 'depth_maps')
os.makedirs(image_path, exist_ok=True)
sparse_path = os.path.join(current_path, 'sparse_original')
os.makedirs(sparse_path, exist_ok=True)
json_path = os.path.join(current_path, 'transforms_train.json')


# ==== 1. Load JSON ====
with open(json_path, "r") as f:
    data = json.load(f)

# 执行转换
filename_mapping = convert_nerf_to_colmap(data, sparse_path)
# 复制深度图
for new_filename, original_filepath in filename_mapping.items():
    # 构造原始文件的完整路径
    original_file_fullpath = os.path.join(image_original_path, original_filepath)
    
    # 构造目标文件的完整路径
    target_file_fullpath = os.path.join(image_path, new_filename)
    
    # 确保目标目录存在
    os.makedirs(os.path.dirname(target_file_fullpath), exist_ok=True)
    
    # 复制文件
    shutil.copy2(original_file_fullpath, target_file_fullpath)

print(f"✅ Images from {original_file_fullpath} copied to: {image_path}")