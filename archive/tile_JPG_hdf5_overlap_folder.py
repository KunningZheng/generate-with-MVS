import h5py
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from deeplsd.geometry.viz_2d import get_flow_vis

def split_and_save_hdf5_jpg(folder_path, output_path, patch_size=(1200, 800), overlap=0.5):
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    jpg_output_dir = os.path.join(output_path, "images", 'train')
    hdf5_output_dir = os.path.join(output_path, "hdf5", 'train')
    overlap_output_dir = os.path.join(output_path, "overlap_mask")
    os.makedirs(jpg_output_dir, exist_ok=True)
    os.makedirs(hdf5_output_dir, exist_ok=True)
    os.makedirs(overlap_output_dir, exist_ok=True)

    # 获取文件夹中的所有文件
    images_dir = os.path.join(folder_path, "images")
    hdf5_dir = os.path.join(folder_path, "hdf5")
    overlap_dir = os.path.join(folder_path, "overlap_mask")
    jpg_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".JPG") or f.endswith(".jpg")])
    hdf5_files = sorted([f for f in os.listdir(hdf5_dir) if f.endswith(".hdf5")])
    overlap_files = sorted([f for f in os.listdir(overlap_dir) if f.endswith(".npy")])

    # 确保文件数量匹配
    if len(jpg_files) != len(hdf5_files) != len(overlap_files):
        raise ValueError("Number of JPG and HDF5 files do not match.")

    # 循环处理每个文件对
    for jpg_file, hdf5_file, overlap_file in zip(jpg_files, hdf5_files, overlap_files):
        jpg_path = os.path.join(images_dir, jpg_file)
        hdf5_path = os.path.join(hdf5_dir, hdf5_file)
        overlap_path = os.path.join(overlap_dir, overlap_file)

        # 读取JPG文件（改为读取RGB图像）
        image = cv2.imread(jpg_path, cv2.IMREAD_COLOR)  # 修改为彩色图
        img_size = np.array(image.shape[:2])  # 获取图像的尺寸
        h, w = img_size

        # 读取HDF5文件
        with h5py.File(hdf5_path, "r") as f:
            df = np.array(f['df']).reshape(img_size)
            line_level = np.array(f['line_level']).reshape(img_size)
            closest = np.array(f['closest']).reshape(h, w, 2)[:, :, [1, 0]]
            bg_mask = np.array(f['bg_mask']).reshape(img_size)
            angle = np.mod(np.array(f['line_level']).reshape(img_size), np.pi)

        # 读取overlap mask
        overlap_mask = np.load(overlap_path)

        # 计算滑动窗口的步长（基于重叠率）
        step_x = int(patch_size[0] * (1 - overlap))
        step_y = int(patch_size[1] * (1 - overlap))

        patch_index = 0

        for x in range(0, w - patch_size[0] + 1, step_x):
            for y in range(0, h - patch_size[1] + 1, step_y):
                # 裁剪overlap mask
                patch_overlap_mask = overlap_mask[y:y + patch_size[1], x:x + patch_size[0]]
                overlap_total_num = np.count_nonzero(patch_overlap_mask > 0)
                ratio = overlap_total_num / (patch_size[0] * patch_size[1])
                # 剔除小于0.99的patch，保证有效训练区域
                if ratio < 0.99:
                    # Test：可视化不符合标准的overlap_mask
                    overlap_patch_path = os.path.join(output_path, f"{overlap_file[:-4]}_patch_{patch_index}.png")
                    patch_overlap_mask_normalized = (patch_overlap_mask * 255).astype(np.uint8)  # 将0和1的矩阵转换为0-255的灰度图
                    cv2.imwrite(overlap_patch_path, patch_overlap_mask_normalized)
                    continue

                # 裁剪图像块（确保裁剪的是RGB图像）
                patch_img = image[y:y + patch_size[1], x:x + patch_size[0]]

                # 裁剪HDF5数据块
                patch_df = df[y:y + patch_size[1], x:x + patch_size[0]]
                patch_line_level = line_level[y:y + patch_size[1], x:x + patch_size[0]]
                patch_closest = closest[y:y + patch_size[1], x:x + patch_size[0]]
                patch_bg_mask = bg_mask[y:y + patch_size[1], x:x + patch_size[0]]

                # 保存JPG小块（确保保存的是RGB图像）
                img_patch_path = os.path.join(jpg_output_dir, f"{jpg_file[:-4]}_patch_{patch_index}.jpg")
                cv2.imwrite(img_patch_path, patch_img)

                # 保存HDF5小块
                hdf5_patch_path = os.path.join(hdf5_output_dir, f"{hdf5_file[:-5]}_patch_{patch_index}.hdf5")
                with h5py.File(hdf5_patch_path, "w") as f:
                    f.create_dataset("df", data=patch_df.flatten())
                    f.create_dataset("line_level", data=patch_line_level.flatten())
                    f.create_dataset("closest", data=patch_closest.flatten())
                    f.create_dataset("bg_mask", data=patch_bg_mask.flatten())

                patch_index += 1

        print(f"Processed {jpg_file} and {hdf5_file}, saved {patch_index} patches.")

# 示例用法
folder_path = "/home/rylynn/Pictures/LinesDetection_Workspace/datasets/Dublin_block1/intermediate_results_0207/gt/"  # 输入文件夹路径
output_path = "/media/rylynn/data/Dublin/block1/gt_patches_dublin_H_matched/"  # 输出文件夹路径
split_and_save_hdf5_jpg(folder_path, output_path, patch_size=(1024, 1024), overlap=0.2)