import os
import cv2
import numpy as np
import h5py
import math

def tile_coordinates(img, patch_size, overlap):
    '''
        计算图像分块的坐标
    '''
    h, w = img.shape[:2]
    stride = patch_size - overlap
    coordinates = []
    
    # 初始化行和列的索引
    row_idx = 0
    
    # 迭代计算分块的起始行 top
    # 循环的范围应该覆盖到图像高度 h
    top_starts = list(range(0, h, stride))
    # 确保最后一个分块能覆盖到图像底部
    if top_starts[-1] + patch_size < h:
        top_starts.append(h - patch_size)
    elif top_starts[-1] + patch_size > h and top_starts[-1] != h - patch_size:
        top_starts[-1] = h - patch_size # 如果最后一个步进起点超出且没有完整覆盖，则回退到恰好覆盖底部
    
    # 处理高度为 h 的图像，且 patch_size > h 的情况
    if h <= patch_size:
        top_starts = [0]


    # 迭代计算分块的起始列 left
    # 循环的范围应该覆盖到图像宽度 w
    left_starts = list(range(0, w, stride))
    # 确保最后一个分块能覆盖到图像右侧
    if left_starts[-1] + patch_size < w:
        left_starts.append(w - patch_size)
    elif left_starts[-1] + patch_size > w and left_starts[-1] != w - patch_size:
        left_starts[-1] = w - patch_size # 如果最后一个步进起点超出且没有完整覆盖，则回退到恰好覆盖右侧
    
    # 处理宽度为 w 的图像，且 patch_size > w 的情况
    if w <= patch_size:
        left_starts = [0]
    
    # 移除重复的边缘起点，这在边界恰好落在步进点时可能发生
    top_starts = sorted(list(set(top_starts)))
    left_starts = sorted(list(set(left_starts)))

    for i, top in enumerate(top_starts):
        col_idx = 0
        for j, left in enumerate(left_starts):
            
            bottom = top + patch_size
            right = left + patch_size
            
            coordinates.append((top, bottom, left, right, i, j))
            col_idx += 1
            
        row_idx += 1 # 实际的行索引在内层循环结束后增加
        
    return coordinates


def tile_and_save_images(images_path, nonoverlap_img_names, images_patches_path, patch_size=2048, overlap=0):
    
    ## Step0: 判断image_list.txt是否存在，存在则跳过分块
    images_list_path = os.path.join(images_patches_path, 'images_patches_list.txt')
    if os.path.exists(images_list_path):
        print(f"{images_list_path} already exists, skipping tiling.")
        return

    ## Step1: 分块影像
    images_patches_list = []
    for img_name in nonoverlap_img_names:
        img_path = os.path.join(images_path, img_name)
        img = cv2.imread(img_path, 1)
        patch_coords = tile_coordinates(img, patch_size, overlap)
        for (top, bottom, left, right, r_idx, c_idx) in patch_coords:
            patch = img[top:bottom, left:right]
            patch_filename = f"{os.path.splitext(img_name.split('/')[-1])[0]}_r{r_idx}_c{c_idx}.jpg"
            patch_path = os.path.join(images_patches_path, patch_filename)
            os.makedirs(images_patches_path, exist_ok=True)
            cv2.imwrite(patch_path, patch)
            images_patches_list.append(patch_path)

    ## Step2: 存储images_patches_list.txt
    with open(images_list_path, 'w') as f_list:
            for patch_path in images_patches_list:
                f_list.write(f"{patch_path}\n")
    print(f"Tiled images saved and list written to {images_list_path}.")
    

def load_tiled_raster_lines(img, img_name, output_path, patch_size, overlap):
    '''
        加载Achor Map中的Raster_lines，合并为一张大图
    '''
    ## Step1: 依据patch_size和overlap，计算分块数量
    patch_coords = tile_coordinates(img, patch_size, overlap)
    raster_lines_global = np.zeros_like(img).astype(float)

    for (top, bottom, left, right, r_idx, c_idx) in patch_coords:
        anchor_map_path = os.path.join(output_path, 'anchor_maps', 
                                        f"{os.path.splitext(img_name)[0]}_r{r_idx}_c{c_idx}.hdf5")
        ## Step2: 读取anchor_map_path中的Raster_lines
        with h5py.File(anchor_map_path, 'r') as hdf5_file:
            bg_mask_block = hdf5_file['bg_mask'][:].reshape((patch_size, patch_size))
            raster_lines_block = 1 - bg_mask_block
        ## Step3: 将读取的Raster_lines放回到raster_lines的对应位置
        raster_lines_global[top:bottom, left:right] += raster_lines_block[0:patch_size, 0:patch_size]

    # Step4: 二值化raster_lines_global,只要出现就认为存在
    raster_lines_global = (raster_lines_global > 0).astype(np.uint8)

    return raster_lines_global


def tiled_df_to_raster_lines(img, img_name, output_path, patch_size, overlap):
    '''
        加载Achor Map中的df，合并为一张大图
    '''
    ## Step1: 依据patch_size和overlap，计算分块数量
    patch_coords = tile_coordinates(img, patch_size, overlap)
    raster_lines_global = np.zeros_like(img).astype(float)

    for (top, bottom, left, right, r_idx, c_idx) in patch_coords:
        anchor_map_path = os.path.join(output_path, 'anchor_maps', 
                                        f"{os.path.splitext(img_name)[0]}_r{r_idx}_c{c_idx}.hdf5")
        
        # Step2: 读取df_block
        with h5py.File(anchor_map_path, 'r') as hdf5_file:
            df_block = hdf5_file['df'][:].reshape((patch_size, patch_size))
        # Step3: 
        raster_lines = (df_block < 2).astype(np.uint8)
        raster_lines_global[top:bottom, left:right] += raster_lines[0:patch_size, 0:patch_size]
    
    # Step4: 二值化得到raster_lines_global
    raster_lines_global = (raster_lines_global>0).astype(np.uint8)

    return raster_lines_global