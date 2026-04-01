import os
import sys
import json
import shutil
import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
import cv2
from functools import reduce
import matplotlib.pyplot as plt
import h5py
from pytlsd import lsd
from afm_op import afm
from collections import defaultdict

# 假设这些是你原本的库，保持不变
from datasets.dataset_reader import load_sparse_model, match_pair, read_depth, compute_bounding_box
from transformation.views_transform_fang import views_transform, grid_reprojection
from deeplsd.datasets.utils.homographies import sample_homography, warp_lines
from deeplsd.geometry.line_utils import clip_line_to_boundaries
# 注意：viz_lines2D2 和 get_flow_vis 可能需要根据分块做调整，这里为了代码稳定性暂时只在生成GT时保留核心逻辑
from deeplsd.geometry.viz_2d import get_flow_vis
from utils.visualize import viz_lines2D2


homography_params = {
    'translation': True,
    'rotation': True,
    'scaling': True,
    'perspective': True,
    'scaling_amplitude': 0.1,
    'perspective_amplitude_x': 0.05,
    'perspective_amplitude_y': 0.05,
    'patch_ratio': 0.95,
    'max_angle': 1.57,
    'allow_artifacts': True
}

def resize_image(img, scale):
    new_height = img.shape[0] // scale
    new_width = img.shape[1] // scale
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img

def get_grid_crops(h, w, size, overlap):
    """
    生成分块坐标
    Returns: list of (r, c, r_start, r_end, c_start, c_end)
             r, c 是块的索引
             *_start, *_end 是像素坐标
    """
    crops = []
    stride = size - overlap
    
    # 计算行起始点
    r_starts = list(range(0, h - size, stride))
    if not r_starts or r_starts[-1] + size < h:
        r_starts.append(max(0, h - size)) # 确保覆盖最后边缘
    
    # 计算列起始点
    c_starts = list(range(0, w - size, stride))
    if not c_starts or c_starts[-1] + size < w:
        c_starts.append(max(0, w - size))
        
    for r_idx, r_start in enumerate(r_starts):
        for c_idx, c_start in enumerate(c_starts):
            r_end = r_start + size
            c_end = c_start + size
            # 修正边缘（虽然上面逻辑已经保证了size大小，但在图像极小情况下需要clip）
            r_end = min(r_end, h)
            c_end = min(c_end, w)
            crops.append((r_idx, c_idx, r_start, r_end, c_start, c_end))
            
    return crops


def visualize_block(out_dir, prefix, df, angle, raster, bg_mask=None):
    """
    可视化分块结果的辅助函数
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Visualize DF (Distance Field)
    # 使用 masked array 处理 NaN，避免可视化全黑或报错
    df_masked = np.ma.masked_invalid(df)
    # 保存时使用 viridis_r colormap (近处亮，远处暗/蓝)
    plt.imsave(os.path.join(out_dir, f'{prefix}_df.jpg'), df_masked, cmap='viridis_r')

    # 2. Visualize Angle (Flow Field)
    try:
        # get_flow_vis 需要填充 NaN，否则可能会报错
        vis_df = df.copy()
        vis_angle = angle.copy()
        vis_df[np.isnan(vis_df)] = np.nanmax(vis_df) if not np.all(np.isnan(vis_df)) else 100.0
        vis_angle[np.isnan(vis_angle)] = 0
        
        angle_field = get_flow_vis(vis_df, vis_angle)
        plt.imsave(os.path.join(out_dir, f'{prefix}_angle.jpg'), angle_field)
    except Exception as e:
        # 极端情况（如全NaN）可能导致可视化失败，打印警告但不中断流程
        # print(f"Warning: Visualizing angle failed for {prefix}: {e}")
        pass

    # 3. Visualize Raster Lines
    plt.imsave(os.path.join(out_dir, f'{prefix}_raster_lines.jpg'), raster, cmap='binary')

    # 4. Visualize Background Mask (if provided)
    if bg_mask is not None:
        plt.imsave(os.path.join(out_dir, f'{prefix}_bg_mask.jpg'), bg_mask, cmap='binary')


def save_homography_lines(out_path, H_list, lines_list):
    num_H = len(H_list)
    max_lines = max([len(l) for l in lines_list]) if lines_list else 0
    if max_lines == 0: max_lines = 1 # Prevent error if empty

    with h5py.File(out_path, 'w') as f:
        f.create_dataset("homographies", data=np.stack(H_list))
        line_counts = np.array([len(l) for l in lines_list], dtype=np.int32)
        f.create_dataset("line_counts", data=line_counts)
        dset = f.create_dataset(
            "lines",
            shape=(num_H, max_lines, 2, 2),
            dtype=np.float32,
            fillvalue=np.nan
        )
        for i, lines in enumerate(lines_list):
            if len(lines) > 0:
                dset[i, :len(lines)] = lines.astype(np.float32)

def Homography_adaptation(cam_id, camerasInfo, homography_outpath, images_path, output_path, num_H):
    # 此函数保持原逻辑不变，生成单应性变换后的线段
    cam_dict = camerasInfo[cam_id]
    img_name = cam_dict['img_name'].split('/')[-1]
    
    if os.path.exists(os.path.join(homography_outpath, img_name+'_homography.hdf5')):
        return

    img = cv2.imread(os.path.join(images_path, cam_dict['img_name']+'.jpg'), 0)
    h, w = img.shape[:2]
    size = (w, h)        

    H_list = []
    lines_list = []
    for i in range(num_H):
        if i == 0:
            H = np.eye(3)
        else:
            H = sample_homography(img.shape, **homography_params)
        H_inv = np.linalg.inv(H)
        H_list.append(H)
        
        warped_img = cv2.warpPerspective(img, H, size, borderMode=cv2.BORDER_REPLICATE)
        warped_lines = lsd(warped_img)[:, [1, 0, 3, 2]].reshape(-1, 2, 2)
        lines = warp_lines(warped_lines, H_inv)
        new_lines, valid = clip_line_to_boundaries(lines, [h,w], min_len=0)
        lines = new_lines[valid]
        lines_list.append(lines)

    save_homography_lines(os.path.join(homography_outpath, img_name+'_homography.hdf5'), H_list, lines_list)    

def Aggregate_by_averaging_tiled(cam_dict, img, depth, ncam_dict, nimg, ndepth, nlines_list, H_list, 
                                 border_margin, min_counts, block_size, overlap):
    """
    分块处理聚合函数。
    先进行全局投影，然后分块计算AFM和统计。
    返回: dict, key=(r_idx, c_idx), value={'df':..., 'angle':..., 'closest':..., 'raster':...}
    """
    h, w = img.shape[:2]
    
    # 1. 生成块坐标
    crops = get_grid_crops(h, w, block_size, overlap)
    
    # 2. 初始化每个块的累加器
    # block_buffers 结构: {(r_idx, c_idx): {'dfs': [], 'angles': [], 'closests': [], 'counts': [], 'raster_accum': ...}}
    block_buffers = {}
    for (r_idx, c_idx, rs, re, cs, ce) in crops:
        bh, bw = re - rs, ce - cs
        block_buffers[(r_idx, c_idx)] = {
            'dfs': [], 'angles': [], 'closests': [], 'counts': [],
            'raster_accum': np.zeros((bh, bw), dtype=np.float32), # 用于累加
            'shape': (bh, bw),
            'offset': (rs, cs),
            'pix_loc': np.stack(np.meshgrid(np.arange(bh), np.arange(bw), indexing='ij'), axis=-1) # Local grid
        }

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_margin * 2, border_margin * 2))

    # 3. 循环单应性变换 (处理最耗时的全局变换)
    for H, nlines in zip(tqdm(H_list, desc="Prepare AF DF in blocks:"), nlines_list):
        # --- 全局操作 (Global Operations) ---
        # 3.1 视角转换 (Lines Transform)
        lines_global = views_transform(img, nlines, cam_dict, nimg, ndepth, ncam_dict)
        
        # 3.2 像素重投影 (Pixel Reprojection Mask)
        count_global = grid_reprojection(nimg, ndepth, ncam_dict, img, depth, cam_dict, H)
        count_global = count_global.astype('uint8')
        count_global = cv2.erode(count_global, kernel) # 全局erode

        # --- 分块操作 (Block Operations) ---
        # 遍历每个块，切分数据并计算局部AFM
        for (r_idx, c_idx, rs, re, cs, ce) in crops:
            buff = block_buffers[(r_idx, c_idx)]
            bh, bw = buff['shape']
            
            # Crop mask
            count_block = count_global[rs:re, cs:ce]
            
            # Shift and Clip lines to block local coordinates
            # 线段坐标减去块的左上角偏移
            lines_block = lines_global.copy()
            lines_block -= np.array([[[rs, cs], [rs, cs]]]) # x, y 顺序: [row, col]
            
            # 裁剪线段到块范围内
            new_lines_block, valid = clip_line_to_boundaries(lines_block, [bh, bw], min_len=1)
            lines_block = new_lines_block[valid]
            # 计算局部 AFM
            num_lines = len(lines_block)
            if num_lines > 0:
                cuda_lines = torch.from_numpy(lines_block[:, :, [1, 0]].astype(np.float32))
                cuda_lines = cuda_lines.reshape(-1, 4)[None].cuda()
                
                # AFM op
                offset = afm(
                    cuda_lines,
                    torch.IntTensor([[0, num_lines, bh, bw]]).cuda(), bh, bw)[0]
                offset = offset[0].permute(1, 2, 0).cpu().numpy()[:, :, [1, 0]]
                
                # 计算局部结果
                closest_local = buff['pix_loc'] + offset
                df_block = np.linalg.norm(offset, axis=-1)
                angle_block = np.mod(np.arctan2(offset[:, :, 0], offset[:, :, 1]) + np.pi / 2, np.pi)
            else:
                # 如果块内没有线段，填充NAN值
                df_block = np.full((bh, bw), np.inf, dtype=np.float32)
                angle_block = np.zeros((bh, bw), dtype=np.float32)
                closest_local = np.zeros((bh, bw, 2), dtype=np.float32)
            
            # 存入 buffer
            buff['dfs'].append(df_block)
            buff['angles'].append(angle_block)
            buff['closests'].append(closest_local)
            buff['counts'].append(count_block)
            
            # Raster accumulation
            buff['raster_accum'] += (df_block < 1).astype(np.uint8) * count_block

    # 4. 聚合每个块的结果 (Compute Medians per block)
    block_results = {}
    
    for key, buff in tqdm(block_buffers.items(), 
                          total=len(block_buffers), 
                          desc="Compute Medians per block"):
        dfs = np.stack(buff['dfs'])
        angles = np.stack(buff['angles'])
        closests = np.stack(buff['closests'])
        counts = np.stack(buff['counts'])
        raster_accum = buff['raster_accum']

        # 计算每个像素的总有效采样次数
        total_counts = np.sum(counts, axis=0)
        # 设定阈值：如果一个像素在 100 次变换中，有效覆盖次数少于 20%，
        # 说明它处于边缘不稳定区域，强制将其视为无效 (NaN)
        valid_ratio_threshold = 1  
        num_H = len(buff['dfs'])
        min_valid_samples = num_H * valid_ratio_threshold
        # 生成高置信度掩码
        confidence_mask = total_counts >= min_valid_samples

        # Median of DF
        dfs[counts == 0] = np.nan
        avg_df = np.nanmedian(dfs, axis=0)
        avg_df[~confidence_mask] = np.nan

        # Median of closest (保持局部坐标)
        closests[counts == 0] = np.nan
        avg_closest = np.nanmedian(closests, axis=0)
        avg_closest[~confidence_mask] = np.nan

        # Median of angle
        circ_bound = (np.minimum(np.pi - angles, angles) * counts).sum(0) / (counts.sum(0) + 1e-6) < 0.3
        angles[:, circ_bound] -= np.where(
            angles[:, circ_bound] > np.pi / 2,
            np.ones_like(angles[:, circ_bound]) * np.pi,
            np.zeros_like(angles[:, circ_bound])
        )
        angles[counts == 0] = np.nan
        avg_angle = np.mod(np.nanmedian(angles, axis=0), np.pi)
        avg_angle[~confidence_mask] = np.nan

        # Raster lines
        raster_lines = np.where(raster_accum > min_counts, np.ones_like(avg_df), np.zeros_like(avg_df))
        raster_lines[~confidence_mask] = 0 # 强制清除低置信度区域的线条
        raster_lines = cv2.dilate(raster_lines, np.ones((5, 5), dtype=np.uint8)) # block较小，kernel适当减小

        block_results[key] = {
            'df': avg_df, 
            'angle': avg_angle, 
            'closest': avg_closest, 
            'raster_lines': raster_lines
        }
        r_idx, c_idx = key
        avg_df_masked = np.ma.masked_invalid(avg_df)  # 创建masked array，自动处理NaN

        plt.imsave(os.path.join('/home/rylynn/Pictures/LinesDetection_Workspace/datasets/Dublin/block2/intermediate_results/avg_blocks/',
                                str(r_idx) + '_' + str(c_idx) + '_df.jpg'), avg_df_masked, cmap='viridis_r')
        plt.imsave(os.path.join('/home/rylynn/Pictures/LinesDetection_Workspace/datasets/Dublin/block2/intermediate_results/avg_blocks/',
                                str(r_idx) + '_' + str(c_idx) + '_raster_lines.jpg'), raster_lines, cmap='binary')
        plt.imsave(os.path.join('/home/rylynn/Pictures/LinesDetection_Workspace/datasets/Dublin/block2/intermediate_results/avg_blocks/',
                                str(r_idx) + '_' + str(c_idx) + '_confidence_mask.jpg'), confidence_mask, cmap='binary', vmin=0, vmax=1)

    return block_results

def main():
    ####################################### 参数设置 #######################################
    workspace = r"/home/rylynn/Pictures/LinesDetection_Workspace/datasets/Dublin/block2/"
    n_jobs = 8
    num_H = 20
    image_scale = 1
    border_margin = 3
    min_counts = 7
    
    # 分块参数
    block_size = 1024
    overlap = 32

    ####################################### 路径设置 #######################################
    sparse_model_path = os.path.join(workspace, 'sparse')
    images_path = os.path.join(workspace, 'images')
    depth_path = os.path.join(workspace, 'depth_maps')
    output_path = os.path.join(workspace, 'intermediate_results')
    
    gt_hdf5_path = os.path.join(workspace, 'gt', 'hdf5')
    gt_img_path = os.path.join(workspace, "gt", "images")
    avg_vis_path = os.path.join(output_path, "avg_blocks")       # 存放中间平均结果的可视化
    final_vis_path = os.path.join(output_path, "visualize_blocks") # 存放最终GT的可视化
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(gt_hdf5_path, exist_ok=True)
    os.makedirs(gt_img_path, exist_ok=True)
    os.makedirs(avg_vis_path, exist_ok=True)
    os.makedirs(final_vis_path, exist_ok=True)

    # 读取sparse model
    camerasInfo, _ = load_sparse_model(sparse_model_path, image_scale)
    for cam_dict in camerasInfo:
        if "points3D_ids" in cam_dict: del cam_dict["points3D_ids"]
        if "points3D_to_xys" in cam_dict: del cam_dict["points3D_to_xys"]

    # 读取 near_image_ids
    json_files = [f for f in os.listdir(output_path) if f.startswith("near_image_ids_") and f.endswith(".json")]
    if len(json_files) == 0:
        print("[ERROR] No near_image_ids_**.json file found.")
        return
    with open(os.path.join(output_path, json_files[0]), "r") as f:
        near_image_ids = json.load(f)

    ####################################### Stage1: 所有航片单应性变换，LSD提取线段并存储 #######################################
    # 统计需要处理的航片
    images_used = set()
    for id, near_images in near_image_ids.items():
        images_used.add(int(id))
        for iid in near_images:
            images_used.add(int(iid))
    homography_outpath = os.path.join(output_path, "homography_lines")
    os.makedirs(homography_outpath,exist_ok=True)
    '''
    Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(Homography_adaptation)(
        cam_id, camerasInfo, homography_outpath, images_path, output_path, num_H)
                                            for cam_id in tqdm(images_used))
    '''
       
    ####################################### Stage2: 逐不重叠航片循环 #######################################
    # 当前视角加载
    for img_id, nimg_ids in near_image_ids.items():
        img_id = int(img_id)
        cam_dict = camerasInfo[img_id]
        img_name = cam_dict['img_name'].split('/')[-1]
        print(f"{img_name} start processing...")
        
        img = cv2.imread(os.path.join(images_path, cam_dict['img_name']+'.jpg'), 0)
        img = resize_image(img, image_scale)
        h, w = img.shape[:2]
        
        depth = read_depth(os.path.join(depth_path, cam_dict['img_name']+'.jpg.geometric.bin'))
        depth = depth / image_scale

        # === 收集所有邻域视角的块结果 ===
        # all_blocks_buffer 结构: {(r_idx, c_idx): {'dfs': [], 'angles': [], 'closests': [], 'raster_lines': []}}
        # 这里的列表存储的是来自不同邻域视角的该块的结果
        all_blocks_buffer = defaultdict(lambda: {'dfs': [], 'angles': [], 'closests': [], 'raster_lines': []})

        for nimg_id in tqdm(nimg_ids, desc=f"Neighbors for {img_name}"):
            ncam_dict = camerasInfo[nimg_id]
            nimg_name = ncam_dict['img_name'].split('/')[-1]
            nimg = cv2.imread(os.path.join(images_path, ncam_dict['img_name']+'.jpg'), 0)
            nimg = resize_image(nimg, image_scale)
            ndepth = read_depth(os.path.join(depth_path, ncam_dict['img_name']+'.jpg.geometric.bin'))
            ndepth = ndepth / image_scale

            # 读取HDF5
            with h5py.File(os.path.join(homography_outpath, nimg_name+'_homography.hdf5'), 'r') as f:
                H_list = f["homographies"][:]
                line_counts = f["line_counts"][:]
                lines_all = f["lines"][:]
            # 恢复每次的真实线段
            nlines_list = [lines_all[i, :line_counts[i]] for i in range(num_H)]
            H_list = [H_list[i] for i in range(num_H)]

            # === 调用分块处理函数 ===
            # 返回的是当前邻域视角 nimg 对当前视角 img 贡献的块结果
            neighbor_blocks = Aggregate_by_averaging_tiled(
                cam_dict, img, depth, ncam_dict, nimg, ndepth, nlines_list, H_list,
                border_margin, min_counts, block_size, overlap
            )
            
            # 将结果按块ID收集起来
            for key, res in neighbor_blocks.items():
                r_idx, c_idx = key
                all_blocks_buffer[key]['dfs'].append(res['df'])
                all_blocks_buffer[key]['angles'].append(res['angle'])
                all_blocks_buffer[key]['closests'].append(res['closest'])
                all_blocks_buffer[key]['raster_lines'].append(res['raster_lines'])
            # 恢复可视化: 这里会生成大量图片 (neighbors * blocks)，建议按需开启
            # 命名格式: img_neighbor_row_col_type.jpg
            prefix = f"{img_name}_{nimg_name}_{r_idx}_{c_idx}"
            # 将结果保存到 avg_blocks 文件夹
            '''
            visualize_block(avg_vis_path, prefix, 
                            res['df'], res['angle'], res['raster_lines'])
            '''                
        
        # === 聚合所有邻域结果，生成 GT 并分块保存 ===
        print(f"Aggregating and saving blocks for {img_name}...")
        
        # [修改点 1] 读取彩色原图并缩放，用于生成分块图片 (替代原来的 shutil.copy)
        # 之前的 img 是灰度图(flag=0)，这里为了保存可视化图片，读取彩色图
        img_color = cv2.imread(os.path.join(images_path, cam_dict['img_name']+'.jpg'))
        if img_color is None:
            print(f"[ERROR] Could not read image: {cam_dict['img_name']}")
            continue
        img_color = resize_image(img_color, image_scale) # 确保尺寸和处理时一致
        
        # [修改点 2] 重新获取分块坐标映射
        # 我们需要知道每个 (r_idx, c_idx) 对应原图的哪个区域 (rs:re, cs:ce)
        # 注意：这里的 h, w 已经在前面定义，是缩放后的高宽
        crops_coords = get_grid_crops(h, w, block_size, overlap)
        # 创建查找表: key=(r_idx, c_idx), value=(rs, re, cs, ce)
        coords_lookup = { (r, c): (rs, re, cs, ce) for r, c, rs, re, cs, ce in crops_coords }

        for key, buffer in all_blocks_buffer.items():
            r_idx, c_idx = key
            
            # 获取该块的像素坐标范围
            if key not in coords_lookup:
                print(f"[WARN] Key {key} not found in crops, skipping.")
                continue
            rs, re, cs, ce = coords_lookup[key]

            # 堆叠所有邻域的结果
            dfs = np.stack(buffer['dfs'])         # (N_neighbors, bh, bw)
            angles = np.stack(buffer['angles'])
            closests = np.stack(buffer['closests'])
            raster_lines_list = buffer['raster_lines']

            # 1. 计算 GT DF (取最小值)
            gt_df = np.nanmin(dfs, axis=0)
            min_index = np.nanargmin(dfs, axis=0) 

            # 2. 根据最小值索引提取对应的 Angle 和 Closest
            rows, cols = np.indices(min_index.shape)
            gt_angle = angles[min_index, rows, cols]
            gt_closest = closests[min_index, rows, cols] # 局部坐标

            # 3. 聚合 Raster Lines (逻辑或)
            gt_raster_lines = reduce(np.logical_or, raster_lines_list).astype(np.uint8)
            gt_raster_lines = cv2.dilate(gt_raster_lines, np.ones((5, 5), dtype=np.uint8))
            gt_bg_mask = (1 - gt_raster_lines).astype(float)

            # [修改点 3] 保存分块图片 (Image Block)
            # 文件名格式: imgname_row_col.jpg
            block_img_name = f"{img_name}_{r_idx}_{c_idx}.jpg"
            img_block = img_color[rs:re, cs:ce] # 裁剪
            cv2.imwrite(os.path.join(gt_img_path, block_img_name), img_block)

            # [可视化2] Final GT Visualization
            # 命名格式: img_row_col_type.jpg
            prefix = f"{img_name}_{r_idx}_{c_idx}"
            visualize_block(final_vis_path, prefix, 
                            gt_df, gt_angle, gt_raster_lines, gt_bg_mask)
            
            # 4. 保存该块的 HDF5 结果
            block_filename = f"{img_name}_{r_idx}_{c_idx}.hdf5"
            block_out_path = os.path.join(gt_hdf5_path, block_filename)
            
            with h5py.File(block_out_path, "w") as f:
                f.create_dataset("df", data=gt_df.flatten())
                f.create_dataset("line_level", data=gt_angle.flatten())
                f.create_dataset("closest", data=gt_closest.flatten())
                f.create_dataset("bg_mask", data=gt_bg_mask.flatten())
                f.attrs['block_idx'] = (r_idx, c_idx)
                f.attrs['global_offset'] = (rs, cs) # 记录该块在原图的起始位置，方便后续恢复
        
        print(f"{img_name} finish.")

if __name__ == "__main__":
    main()