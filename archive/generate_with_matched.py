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
import multiprocessing
import h5py

from datasets.dataset_reader import (
    load_sparse_model,
    match_pair,
    read_depth
)
from datasets.line3dpp_loader import parse_line_segments, parse_lines3dpp, save_segments_l3dpp
from deeplsd.geometry.line_utils import clip_line_to_boundaries
from transformation.views_transform_fang import views_transform_reverse
from utils.line_tools import establish_line_correspondences_reverse, lsd_opencv, af_df_producer
from utils.visualize import viz_lines2D2
from deeplsd.geometry.viz_2d import get_flow_vis


# --- 新增：Worker 函数，只处理单张图片 ---
def process_single_view_worker(args):
    """
    为了适配 multiprocessing，将所有参数打包到一个元组 args 中，
    或者使用 partial 固定固定参数。
    """
    (img_id, nimg_ids, camerasInfo, images_path, depth_path, output_path) = args

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
    lines = lsd_opencv(img)[:, [1, 0, 3, 2]].reshape(-1, 2, 2)
    lines, valid = clip_line_to_boundaries(lines, img.shape, min_len=0)
    lines = lines[valid]
    viz_lines2D2(img, lines, os.path.join(output_path, 'visualize'), f"{os.path.splitext(img_name)[0]}")
    
    valid_idx_all_local = []

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
        
        nlines = lsd_opencv(nimg)[:, [1, 0, 3, 2]].reshape(-1, 2, 2)
        nlines, valid = clip_line_to_boundaries(nlines, nimg.shape, min_len=0)
        nlines = nlines[valid]

        lines_proj, line_indices = views_transform_reverse(nimg, lines, ncam_dict, img, depth, cam_dict)

        matches = establish_line_correspondences_reverse(lines_proj, nlines, t_angle=1.0, 
                        t_dist=1.0, 
                        t_overlap=0.95)

        valid_idx = set()
        for l_idx_proj, _, _ in matches:
            original_idx = line_indices[l_idx_proj]
            valid_idx.add(original_idx)
        valid_idx_all_local.append(sorted(list(valid_idx)))

    # 统计逻辑
    from collections import Counter
    all_indices = [idx for sublist in valid_idx_all_local for idx in sublist]
    index_counts_counter = Counter(all_indices)
    
    # 筛选至少匹配 1 次的线段
    final_valid_indices = [idx for idx, count in index_counts_counter.items() if count >= 1]
    print(f"Image {img_name}: Total matched lines from neighbors: {len(final_valid_indices)} / {len(lines)}")
    
    # 可视化与保存
    lines_matched = lines[final_valid_indices]
    viz_lines2D2(img, lines_matched, os.path.join(output_path, 'visualize'), f"{os.path.splitext(img_name)[0]}_matched")
    
    lines_matched_l3dpp = lines_matched.reshape(-1,4)[:,[1,0,3,2]]
    save_segments_l3dpp(lines_matched_l3dpp, os.path.join(output_path, 'matched_lines'), img_id+1, width, height)
    
    ############################ 保存为栅格化的形式 ############################
    df, angle, closest, raster_lines = af_df_producer(lines_matched, img)
    # 读取邻近视角overlap mask
    overlap_mask_path = os.path.join(output_path, "overlap_mask", f"overlap_mask_{img_id}.npy")
    overlap_mask = np.load(overlap_mask_path)
    # 将overlap_mask加入raster_lines，构成训练的背景掩膜
    raster_lines = np.where(overlap_mask == True, raster_lines,
                            np.zeros_like(img))
    raster_lines = cv2.dilate(raster_lines, np.ones((21, 21), dtype=np.uint8))
    bg_mask = (1 - raster_lines).astype(float)

    # Save the DF in a hdf5 file
    out_path = os.path.join(output_path, "hdf5", img_name) + '.hdf5'
    with h5py.File(out_path, "w") as f:
        f.create_dataset("df", data=df.flatten())
        f.create_dataset("line_level", data=angle.flatten())
        f.create_dataset("closest", data=closest.flatten())
        f.create_dataset("bg_mask", data=bg_mask.flatten())

    # visualize
    plt.imsave(os.path.join(output_path, 'visualize', img_name+ '_df.jpg'), df, cmap='viridis_r')
    angle_field= get_flow_vis(df, angle)
    plt.imsave(os.path.join(output_path, 'visualize', img_name+ '_angle.jpg'), angle_field)
    plt.imsave(os.path.join(output_path, 'visualize', img_name+ '_bg_mask.jpg'), bg_mask, cmap='binary')
    return img_id, final_valid_indices

# --- 修改后的主调用函数 ---
def project_and_establish_correspondences_parallel(camerasInfo, overlap_images, images_path, depth_path, output_path):
    
    os.makedirs(os.path.join(output_path, 'matched_lines'), exist_ok=True)
    os.makedirs(os.path.join(output_path, "hdf5"), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'visualize'), exist_ok=True)
    
    valid_idx_images_all = {}
    
    # 准备任务参数列表
    tasks = []
    for img_id, nimg_ids in overlap_images.items():
        # 这里把所有需要的参数打包
        # 注意：camerasInfo 这种大字典在 fork 模式下（Linux默认）是写时复制的，内存开销还好
        # 但如果很大，传递给 spawn 模式的进程会很慢
        tasks.append((
            img_id, nimg_ids, camerasInfo, images_path, depth_path, output_path
        ))

    # 设置并行核心数，建议留几个核心给系统，或者根据内存大小限制
    # 假设你有 3090Ti 这种级别的机器，内存如果 >= 64GB，可以开 8-10 个
    num_processes = min(multiprocessing.cpu_count(), 8) 
    
    print(f"[INFO] Starting parallel processing with {num_processes} processes...")

    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=num_processes) as pool:
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
    # [修改] 强制设置启动方法，防止有些库在 import 时就初始化 CUDA
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    ####################################### 参数 #######################################
    workspace = r"/home/rylynn/Pictures/LinesDetection_Workspace/datasets/Dublin/block2/"

    ####################################### 路径 #######################################
    sparse_model_path = os.path.join(workspace, 'sparse')
    output_path = os.path.join(workspace, 'intermediate_results_0118')


    # 0. 数据准备：读取稀疏模型
    camerasInfo, points_in_images = load_sparse_model(sparse_model_path, image_scale=1)
    print(f"[INFO] Loaded {len(camerasInfo)} images.")

    # 2. 读取不重叠相片及其邻近视角
    json_files = [f for f in os.listdir(output_path) if f.startswith("near_image_ids_") and f.endswith(".json")]
    with open(os.path.join(output_path, json_files[0]), "r") as f:
        near_image_ids = json.load(f)
    # 确保 key 是整数
    near_image_ids = {int(k):v for k,v in near_image_ids.items()}

    # 注意：cv2 在多进程内部可能会再次尝试并行，导致 cpu 争用变慢。
    # 建议在主程序入口处强制 cv2 使用单线程（让进程级并行来利用多核）
    cv2.setNumThreads(0)
    # 3. 根据邻近视角，寻找有匹配的线段
    project_and_establish_correspondences_parallel(
        camerasInfo, near_image_ids,
        images_path=os.path.join(workspace, 'images'),
        depth_path=os.path.join(workspace, 'depth_maps'),
        output_path=output_path,
    )
