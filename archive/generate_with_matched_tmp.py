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


# --- 修改：Worker 函数 ---
def process_single_view_worker(args):
    """
    为了适配 multiprocessing，将所有参数打包到一个元组 args 中。
    """
    (img_id, nimg_ids, camerasInfo, images_path, depth_path, output_path) = args

    # [重要] 防止 matplotlib 在多进程中崩溃
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg') 
    
    img_id = int(img_id)
    
    cam_dict = camerasInfo[img_id]
    img_name = cam_dict['img_name'].split('/')[-1]
    
    # ------------------ [修改点 1] Double Check ------------------
    # 虽然主函数会过滤，但在worker里再判断一次更加稳健
    out_hdf5_path = os.path.join(output_path, "hdf5", img_name) + '.hdf5'
    if os.path.exists(out_hdf5_path):
        return  # 如果存在，直接退出，不做任何计算
    # -----------------------------------------------------------

    width = int(cam_dict['width'])
    height = int(cam_dict['height'])

    # 读取数据
    if os.path.exists(os.path.join(images_path, cam_dict['img_name']+'.png')):
        img = cv2.imread(os.path.join(images_path, cam_dict['img_name']+'.png'), 0)
    else:
        img = cv2.imread(os.path.join(images_path, cam_dict['img_name']+'.jpg'), 0)
    
    # 注意：这里你需要确保 matched_lines_path 这个路径在 worker 里是正确的
    # 建议最好也通过 args 传进来，目前保持原样
    matched_lines_path = '/home/rylynn/Pictures/LinesDetection_Workspace/datasets/Dublin_block1/intermediate_results_matched1/matched_lines/'
    lines_matched = parse_line_segments(matched_lines_path, img_id+1, width, height).reshape(-1, 2, 2)
    
    # 可视化部分（如果在已有文件时跳过，这里也会跳过，节省大量IO时间）
    viz_lines2D2(img, lines_matched, os.path.join(output_path, 'visualize'), f"{os.path.splitext(img_name)[0]}_matched")
    
    ############################ 保存为栅格化的形式 ############################
    df, angle, closest, raster_lines = af_df_producer(lines_matched, img)
    
    # 读取邻近视角overlap mask
    overlap_mask_path = os.path.join(output_path, "overlap_mask", f"overlap_mask_{img_id}.npy")
    
    # 增加一点鲁棒性，防止mask文件不存在报错
    if os.path.exists(overlap_mask_path):
        overlap_mask = np.load(overlap_mask_path)
        # 将overlap_mask加入raster_lines
        raster_lines = np.where(overlap_mask == True, raster_lines, np.zeros_like(img))
    else:
        # 如果没有mask，根据你的逻辑决定是否需要 warning
        # print(f"Warning: Mask not found for {img_id}")
        pass

    raster_lines = cv2.dilate(raster_lines, np.ones((21, 21), dtype=np.uint8))
    bg_mask = (1 - raster_lines).astype(float)

    # Save the DF in a hdf5 file
    # 使用之前定义的 out_hdf5_path
    with h5py.File(out_hdf5_path, "w") as f:
        f.create_dataset("df", data=df.flatten())
        f.create_dataset("line_level", data=angle.flatten())
        f.create_dataset("closest", data=closest.flatten())
        f.create_dataset("bg_mask", data=bg_mask.flatten())

    # visualize
    plt.imsave(os.path.join(output_path, 'visualize', img_name+ '_df.jpg'), df, cmap='viridis_r')
    angle_field= get_flow_vis(df, angle)
    plt.imsave(os.path.join(output_path, 'visualize', img_name+ '_angle.jpg'), angle_field)
    plt.imsave(os.path.join(output_path, 'visualize', img_name+ '_bg_mask.jpg'), bg_mask, cmap='binary')


# --- 修改后的主调用函数 ---
def project_and_establish_correspondences_parallel(camerasInfo, overlap_images, images_path, depth_path, output_path):
    
    os.makedirs(os.path.join(output_path, 'matched_lines'), exist_ok=True)
    os.makedirs(os.path.join(output_path, "hdf5"), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'visualize'), exist_ok=True)
    
    # 准备任务参数列表
    tasks = []
    skipped_count = 0
    
    print("[INFO] Preparing tasks...")
    for img_id, nimg_ids in overlap_images.items():
        
        # ------------------ [修改点 2] 核心优化 ------------------
        # 在加入任务队列前，先检查输出文件是否存在
        cam_dict = camerasInfo[img_id]
        img_name = cam_dict['img_name'].split('/')[-1]
        target_hdf5 = os.path.join(output_path, "hdf5", img_name) + '.hdf5'
        
        if os.path.exists(target_hdf5):
            skipped_count += 1
            continue # 如果文件存在，直接跳过，不放入 tasks 列表
        # -------------------------------------------------------

        tasks.append((
            img_id, nimg_ids, camerasInfo, images_path, depth_path, output_path
        ))

    print(f"[INFO] Total images: {len(overlap_images)}")
    print(f"[INFO] Skipped (Already exist): {skipped_count}")
    print(f"[INFO] Tasks to process: {len(tasks)}")

    if len(tasks) == 0:
        print("[INFO] All tasks completed. Nothing to do.")
        return

    # 设置并行核心数
    num_processes = min(multiprocessing.cpu_count(), 8)
    
    print(f"[INFO] Starting parallel processing with {num_processes} processes...")

    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=num_processes) as pool:
        _ = list(tqdm(pool.imap_unordered(process_single_view_worker, tasks), total=len(tasks), desc="Parallel Correspondence"))

    print("[INFO] Processing completed.")
    return None


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    ####################################### 参数 #######################################
    workspace = r"/home/rylynn/Pictures/LinesDetection_Workspace/datasets/Dublin_block1/"

    ####################################### 路径 #######################################
    sparse_model_path = os.path.join(workspace, 'sparse')
    output_path = os.path.join(workspace, 'intermediate_results_0207')

    # 0. 数据准备：读取稀疏模型
    camerasInfo, points_in_images = load_sparse_model(sparse_model_path, image_scale=1)
    print(f"[INFO] Loaded {len(camerasInfo)} images.")

    # 2. 读取不重叠相片及其邻近视角
    json_files = [f for f in os.listdir(output_path) if f.startswith("near_image_ids_") and f.endswith(".json")]
    if not json_files:
        print("[ERROR] No near_image_ids_*.json found!")
        sys.exit(1)
        
    with open(os.path.join(output_path, json_files[0]), "r") as f:
        near_image_ids = json.load(f)
    near_image_ids = {int(k):v for k,v in near_image_ids.items()}

    # 新增：block1中有些图像植被占地比较多，手动去掉了gt图，所以更新一下near_images_ids
    gt_images_path = os.path.join(output_path, 'gt', 'images')
    near_image_ids_filtered = {}
    for img_id in near_image_ids:
        img_name = camerasInfo[img_id]['img_name'].split('/')[1]
        gt_img_pth = os.path.join(gt_images_path, img_name+'.jpg')
        if os.path.exists(gt_img_pth):
            near_image_ids_filtered[img_id] = near_image_ids[img_id]
            overlap_mask_pth = os.path.join(output_path, "overlap_mask", f"overlap_mask_{img_id}.npy")
            new_overlap_mask_pth = os.path.join(output_path, "gt", "overlap_mask", f"overlap_mask_{img_id}.npy")
            if os.path.exists(overlap_mask_pth):
                # 复制overlap_mask到新路径
                os.makedirs(os.path.dirname(new_overlap_mask_pth), exist_ok=True)
                np.save(new_overlap_mask_pth, np.load(overlap_mask_pth))




    cv2.setNumThreads(0)
    
    # 3. 执行
    project_and_establish_correspondences_parallel(
        camerasInfo, near_image_ids_filtered,
        images_path=os.path.join(workspace, 'images'),
        depth_path=os.path.join(workspace, 'depth_maps'),
        output_path=output_path,
    )