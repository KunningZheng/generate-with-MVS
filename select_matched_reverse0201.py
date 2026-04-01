import os
import json
import numpy as np
from tqdm import tqdm
import random
import networkx as nx
import matplotlib.pyplot as plt
import cv2
import sys
import logging

from datasets.dataset_reader import (
    load_sparse_model,
    match_pair,
    find_common_points,
    compute_bounding_box,
    read_depth
)
from datasets.line3dpp_loader import parse_line_segments, parse_lines3dpp, save_segments_l3dpp
from deeplsd.geometry.line_utils import clip_line_to_boundaries
from transformation.views_transform_fang import views_transform_reverse
from utils.line_tools import establish_line_correspondences_reverse, lsd_opencv
from utils.visualize import viz_lines2D2, viz_pairwise_matches


def compute_overlap_ratio(cam_dict1, cam_dict2, common_points):
    """计算重叠比例"""
    bb_area1 = compute_bounding_box(common_points[:, 0])
    bb_area2 = compute_bounding_box(common_points[:, 1])
    area1 = cam_dict1['width'] * cam_dict1['height']
    area2 = cam_dict2['width'] * cam_dict2['height']
    return bb_area1 / area1, bb_area2 / area2


def save_dict_to_json(data, save_path):
    """
    安全地将包含整型键的字典保存为 JSON
    """
    # 转换所有键为字符串，确保值是原生 Python 类型（非 NumPy 类型）
    serializable_data = {str(k): [int(i) for i in v] for k, v in data.items()}
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=4)


# 1. 所有相片寻找邻近视角
def select_nearby_views(camerasInfo, points_in_images, overlap_percentile = 90):
    '''
    寻找邻近视角
    inputs:
        - camerasInfo: 相机信息字典
        - points_in_images: 图像中3D点索引
        - overlap_percentile: 自动阈值分位数,可以简单理解为取前%为重叠航片，默认90
    outputs:
        - overlap_images: 字典，键为图像ID，值为邻近视角ID列表
    '''
    ######################## Step1:计算动态阈值 ########################
    print("[INFO] Computing match statistics...")
    match_counts = []
    # 计算匹配点矩阵
    matches_matrix,_ = match_pair(camerasInfo, points_in_images)
    # 统计匹配点数量（剔除0）
    match_counts = matches_matrix[matches_matrix > 0].flatten()
    # 根据匹配点分布自适应确定阈值
    match_point_num = int(np.percentile(match_counts, 100-overlap_percentile))  # 等效于从大到小取
    print(f"[INFO] Adaptive match threshold = {match_point_num}")    

    ######################## Step2:根据阈值寻找邻近相片 ########################
    _, overlap_images = match_pair(camerasInfo, points_in_images, match_point_num=match_point_num)

    '''
    # 添加筛选步骤
    area_ratio_th = 0.5           # 面积重叠比例阈值
    dist_th = 25.0                # 相机中心距离阈值（米）
    near_image_ids = {}
    for img1_id in range(len(camerasInfo)):
        cam_dict1 = camerasInfo[img1_id]
        lens1 = cam_dict1['img_name'].split('/')[0]
        near_images = []
        for img2_id in overlap_images.get(img1_id, []):
            cam_dict2 = camerasInfo[img2_id]
            lens2 = cam_dict2['img_name'].split('/')[0]
            dist = np.linalg.norm(np.array(cam_dict1['position']) - np.array(cam_dict2['position']))
            if dist > dist_th:
                continue
            common_points = find_common_points(img1_id, img2_id, camerasInfo)
            if common_points is None or len(common_points) < 10:
                continue
            r1, r2 = compute_overlap_ratio(cam_dict1, cam_dict2, common_points)
            # 保证两个相片的镜头相同（避免大的偏移）
            if r1 > area_ratio_th and r2 > area_ratio_th and lens1 == lens2:
                near_images.append(int(img2_id))
        near_image_ids[img1_id] = near_images
    '''
    json_path = os.path.join(output_path, f"near_image_ids_{match_point_num}_test.json")
    with open(json_path, 'w') as f:
        json.dump(overlap_images, f, indent=2)
    print(f"[INFO] Saved near-image dictionary to {json_path}")
    return overlap_images


import multiprocessing
from functools import partial

# --- 新增：Worker 函数，只处理单张图片 ---
def process_single_view_worker(args):
    """
    为了适配 multiprocessing，将所有参数打包到一个元组 args 中，
    或者使用 partial 固定固定参数。
    """
    (img_id, nimg_ids, camerasInfo, 
     lsd_lines_path, images_path, depth_path, output_path, neighbor_thred) = args

    # [重要] 如果 viz_lines2D2 用到了 plt，必须加上这一句防止崩溃
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg') 
    
    img_id = int(img_id)
    
    # --- 原有逻辑开始 ---
    # 稍微调整：不再需要循环 overlap_images，只处理当前的 img_id
    
    cam_dict = camerasInfo[img_id]
    img_name = cam_dict['img_name'].split('/')[-1]
    width = int(cam_dict['width'])
    height = int(cam_dict['height'])
    
    # 读取数据
    # 确定是.png还是.jpg
    if os.path.exists(os.path.join(images_path, cam_dict['img_name']+'.png')):
        img = cv2.imread(os.path.join(images_path, cam_dict['img_name']+'.png'), 0)
        depth = read_depth(os.path.join(depth_path, cam_dict['img_name']+'.png.geometric.bin'))
    else:
        img = cv2.imread(os.path.join(images_path, cam_dict['img_name']+'.jpg'), 0)
        depth = read_depth(os.path.join(depth_path, cam_dict['img_name']+'.jpg.geometric.bin'))
    
    # 提取当前视角 lines
    #lines = parse_line_segments(lsd_lines_path, img_id+1, width, height).reshape(-1, 2, 2)
    lines = lsd_opencv(img)[:, [1, 0, 3, 2]].reshape(-1, 2, 2)
    lines, valid = clip_line_to_boundaries(lines, img.shape, min_len=0)
    lines = lines[valid]
    
    valid_idx_all_local = [] # 变量名微调，避免混淆

    # 邻近视角循环 (这个内循环保留，因为是在单张图片处理逻辑内)
    for nimg_id in nimg_ids:
        if int(nimg_id) == int(img_id): continue
        
        ncam_dict = camerasInfo[nimg_id]
        nwidth = int(ncam_dict['width'])
        nheight = int(ncam_dict['height'])
        # 确定是.png还是.jpg
        if os.path.exists(os.path.join(images_path, cam_dict['img_name']+'.png')):
            nimg = cv2.imread(os.path.join(images_path, ncam_dict['img_name']+'.png'), 0)
        else:
            nimg = cv2.imread(os.path.join(images_path, ncam_dict['img_name']+'.jpg'), 0)
        
        #nlines = parse_line_segments(lsd_lines_path, nimg_id+1, nwidth, nheight).reshape(-1, 2, 2)
        nlines = lsd_opencv(nimg)[:, [1, 0, 3, 2]].reshape(-1, 2, 2)
        nlines, valid = clip_line_to_boundaries(nlines, nimg.shape, min_len=0)
        nlines = nlines[valid]

        lines_proj, line_indices = views_transform_reverse(nimg, lines, ncam_dict, img, depth, cam_dict)

        matches = establish_line_correspondences_reverse(lines_proj, nlines, t_angle=1.0, 
                        t_dist=1.0, 
                        t_overlap=0.95)

        # --- 新增：准备可视化所需的数据 ---
        viz_pairs = []
        valid_idx = set()
        
        # 解析 matches
        for match_item in matches:
            l_idx_proj = match_item[0]  # line_idx
            n_idx = match_item[1]  # nline_idx
            
            # 将投影索引映射回原始 img 的线段索引
            original_idx = line_indices[l_idx_proj]
            
            valid_idx.add(original_idx)
            
            # 添加到可视化列表 (src_idx, neighbor_idx)
            viz_pairs.append((original_idx, n_idx))
            
        valid_idx_all_local.append(sorted(list(valid_idx)))

        # --- 新增：调用可视化函数 ---
        # 仅当有匹配时才保存，防止生成过多空图
        if len(viz_pairs) > 0:
            viz_name = f"{os.path.splitext(img_name)[0]}_vs_{os.path.splitext(ncam_dict['img_name'].split('/')[-1])[0]}"
            viz_pairwise_matches(
                img, lines,       # 左图和左图线段
                nimg, nlines,     # 右图和右图线段
                viz_pairs,        # 匹配对索引列表
                os.path.join(output_path, "pairwise_viz"), # 建议存放在子文件夹
                viz_name
            )

        valid_idx = set()
        for l_idx_proj, _, _ in matches:
            original_idx = line_indices[l_idx_proj]
            valid_idx.add(original_idx)
        valid_idx_all_local.append(sorted(list(valid_idx)))

    # 统计逻辑
    from collections import Counter
    all_indices = [idx for sublist in valid_idx_all_local for idx in sublist]
    index_counts_counter = Counter(all_indices)
    
    # 筛选至少匹配 3 次的线段 (原代码逻辑)
    final_valid_indices = [idx for idx, count in index_counts_counter.items() if count >= 1]

    # 可视化与保存
    lines_matched = lines[final_valid_indices]
    viz_lines2D2(img, lines_matched, output_path, f"{os.path.splitext(img_name)[0]}_matched")
    
    matched_lines_path = os.path.join(output_path, 'matched_lines') # 需确保路径已创建
    save_segments_l3dpp(lines_matched.reshape(-1,4)[:,[1,0,3,2]], matched_lines_path, img_id+1, width, height)

    # 返回结果：(img_id, 结果列表)
    return img_id, final_valid_indices

# --- 修改后的主调用函数 ---
def project_and_establish_correspondences_parallel(camerasInfo, overlap_images,
                                          lsd_lines_path, images_path, depth_path, output_path, neighbor_thred=8):
    
    matched_lines_path = os.path.join(output_path, 'matched_lines')
    os.makedirs(matched_lines_path, exist_ok=True)
    
    valid_idx_images_all = {}
    
    # 准备任务参数列表
    tasks = []
    for img_id, nimg_ids in overlap_images.items():
        # 这里把所有需要的参数打包
        # 注意：camerasInfo 这种大字典在 fork 模式下（Linux默认）是写时复制的，内存开销还好
        # 但如果很大，传递给 spawn 模式的进程会很慢
        tasks.append((
            img_id, nimg_ids, camerasInfo,
            lsd_lines_path, images_path, depth_path, output_path, neighbor_thred
        ))

    # 设置并行核心数，建议留几个核心给系统，或者根据内存大小限制
    # 假设你有 3090Ti 这种级别的机器，内存如果 >= 64GB，可以开 8-10 个
    num_processes = max(1, 1) #multiprocessing.cpu_count() - 4
    
    print(f"[INFO] Starting parallel processing with {num_processes} processes...")

    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用 imap_unordered 可以实时获取结果，配合 tqdm 显示进度
        results = list(tqdm(pool.imap_unordered(process_single_view_worker, tasks), total=len(tasks), desc="Parallel Correspondence"))

    # 汇总结果
    print("[INFO] Aggregating results...")
    for img_id, valid_indices in results:
        valid_idx_images_all[img_id] = valid_indices

    # 原代码返回了 index_counts，但原代码逻辑里 index_counts 是最后一次循环的局部变量
    # 这里我们暂且忽略它，或者只返回最后处理的一个，通常 valid_idx_images_all 才是重点
    return valid_idx_images_all


if __name__ == "__main__":
    ####################################### 参数 #######################################
    workspace = r"/home/rylynn/Pictures/LinesDetection_Workspace/datasets/Dublin/block2/"  # /home/rylynn/Pictures/LinesDetection_Workspace/datasets/Dublin_block1/
    overlap_percentile = 50       # 自动阈值分位数,可以简单理解为取前%为重叠航片

    ####################################### 路径 #######################################
    sparse_model_path = os.path.join(workspace, 'sparse')
    output_path = os.path.join(workspace, 'intermediate_results_0201')
    line3dpp_path = os.path.join(output_path, 'lsd_lines_len3000')


    # 0. 数据准备：读取稀疏模型
    camerasInfo, points_in_images = load_sparse_model(sparse_model_path, image_scale=1)
    print(f"[INFO] Loaded {len(camerasInfo)} images.")


    # 2. 所有相片寻找邻近视角
    overlap_images = select_nearby_views(camerasInfo, points_in_images, overlap_percentile)

    # 注意：cv2 在多进程内部可能会再次尝试并行，导致 cpu 争用变慢。
    # 建议在主程序入口处强制 cv2 使用单线程（让进程级并行来利用多核）
    cv2.setNumThreads(0)
    # 3. 根据邻近视角，寻找有4次及以上匹配的线段
    corres_idx_all = project_and_establish_correspondences_parallel(
        camerasInfo, overlap_images,
        lsd_lines_path=os.path.join(output_path, 'deeplsd_Dublin_H'),
        images_path=os.path.join(workspace, 'images'),
        depth_path=os.path.join(workspace, 'depth_maps'),
        output_path=output_path,
        neighbor_thred=8
    )
    save_dict_to_json(corres_idx_all, os.path.join(output_path, 'corres1_idx_all.json'))

