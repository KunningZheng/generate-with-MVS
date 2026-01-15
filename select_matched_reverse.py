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
    read_depth,
    load_depth_float32
)
from datasets.line3dpp_loader import parse_line_segments, parse_lines3dpp, save_segments_l3dpp
from deeplsd.geometry.line_utils import clip_line_to_boundaries
from transformation.views_transform_fang import views_transform_reverse
from utils.line_tools import establish_line_correspondences_reverse
from utils.visualize import viz_lines2D2


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
    return overlap_images


def project_and_establish_correspondences(camerasInfo, overlap_images, reconstructed_idx_all, top3000_indices_all,
                                          lsd_lines_path, images_path, depth_path, output_path, neighbor_thred=8):
    
    valid_idx_images_all = {}
    matched_lines_path = os.path.join(output_path, 'matched_lines')
    os.makedirs(matched_lines_path, exist_ok=True)

    '''
    # 配置 logging
    log_file_path = os.path.join(output_path, 'log.txt')

    # 获取根日志记录器
    logger = logging.getLogger()
    # 清理旧的 handler，防止重复写入或配置不生效
    if logger.hasHandlers():
        logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler() # 同时输出到控制台
        ]
    )
    '''
    # 2. 对每个相片，投影邻近视角的线段到当前视角
    for img_id, nimg_ids in tqdm(overlap_images.items(), total=len(overlap_images), desc="Correspondence Establishment"):
        '''
        if len(nimg_ids) < neighbor_thred:
            print(f"[WARNING] Image ID {img_id} has less than {neighbor_thred} neighbors, skipping.")
            continue
        '''

        img_id = int(img_id)
        cam_dict = camerasInfo[img_id]
        img_name = cam_dict['img_name'].split('/')[-1]
        width = int(cam_dict['width'])
        height = int(cam_dict['height'])
        # 确定是.png还是.jpg
        if os.path.exists(os.path.join(images_path, cam_dict['img_name']+'.png')):
            img = cv2.imread(os.path.join(images_path, cam_dict['img_name']+'.png'), 0)
            depth = read_depth(os.path.join(depth_path, cam_dict['img_name']+'.png.geometric.bin'))
        else:
            img = cv2.imread(os.path.join(images_path, cam_dict['img_name']+'.jpg'), 0)
            depth = read_depth(os.path.join(depth_path, cam_dict['img_name']+'.jpg.geometric.bin'))

        
        ## 提取当前视角lines
        lines = parse_line_segments(lsd_lines_path, img_id+1, width, height).reshape(-1, 2, 2)
        lines, valid = clip_line_to_boundaries(lines, img.shape, min_len=0)
        lines = lines[valid]
        viz_lines2D2(img, lines, output_path, f"{os.path.splitext(img_name)[0]}")

        ## 逐邻近视角循环
        valid_idx_all = []
        for nimg_id in nimg_ids:
            # 跳过当前视角
            if int(nimg_id) == int(img_id):
                continue
            ncam_dict = camerasInfo[nimg_id]
            nimg_name = ncam_dict['img_name'].split('/')[-1]
            nwidth = int(ncam_dict['width'])
            nheight = int(ncam_dict['height'])
            # 确定是.png还是.jpg
            if os.path.exists(os.path.join(images_path, cam_dict['img_name']+'.png')):
                nimg = cv2.imread(os.path.join(images_path, ncam_dict['img_name']+'.png'), 0)
            else:
                nimg = cv2.imread(os.path.join(images_path, ncam_dict['img_name']+'.jpg'), 0)
            
            ## 提取邻近视角lines
            nlines = parse_line_segments(lsd_lines_path, nimg_id+1, nwidth, nheight).reshape(-1, 2, 2)
            nlines, valid = clip_line_to_boundaries(nlines, nimg.shape, min_len=0)
            nlines = nlines[valid]

            ## 投影线段，裁剪到邻近视角范围内
            lines_proj, line_indices = views_transform_reverse(nimg, lines, ncam_dict, img, depth, cam_dict)
            #viz_lines2D2(nimg, lines_proj, output_path, f"{os.path.splitext(img_name)[0]}_{nimg_id}")

            # 3. 建立线段对应关系
            matches = establish_line_correspondences_reverse(lines_proj, nlines, t_angle=1.0, 
                                   t_dist=1.0, 
                                   t_overlap=0.95)

            valid_idx = set()
            for l_idx_proj, _, _ in matches:
                # l_idx_proj 是 lines_proj 中的下标 (0 ~ len(lines_proj)-1)
                # 我们需要通过 line_indices 数组找到它在原始 lines 中的真实下标
                original_idx = line_indices[l_idx_proj]
                valid_idx.add(original_idx)

            # 转为列表并排序（保持顺序一致性）
            valid_idx = sorted(list(valid_idx))
            #viz_lines2D2(img, lines[valid_idx], output_path, f"{os.path.splitext(img_name)[0]}_matched_{nimg_id}")
            valid_idx_all.append(valid_idx)

        # 4. 统计每个线段被匹配的次数，记录至少被4个邻近视角匹配到的线段
        from collections import Counter
        all_indices = [idx for sublist in valid_idx_all for idx in sublist]
        index_counts = Counter(all_indices)
        valid_idx_all = [idx for idx, count in index_counts.items() if count >= 1]
        ##logging.info(f"Image {img_name}: Total matched lines from neighbors: {len(valid_idx_all)} / {len(lines)}")
        #viz_lines2D2(img, lines[list(valid_idx_all)], output_path, f"{os.path.splitext(img_name)[0]}_matched")

        valid_idx_images_all[img_id] = valid_idx_all
        lines_matched = lines[valid_idx_all]
        viz_lines2D2(img, lines_matched, output_path, f"{os.path.splitext(img_name)[0]}_matched")

        # 存储匹配线段
        save_segments_l3dpp(lines_matched.reshape(-1,4)[:,[1,0,3,2]], matched_lines_path, img_id+1, width, height)

    return index_counts, valid_idx_images_all

# 6. 对比各结果
def analyze_results(index_counts, corres_idx_all, reconstructed_idx_all, top3000_indices_all, images_path, lsd_lines_path, output_path):
    # 配置 logging
    log_file_path = os.path.join(output_path, 'log.txt')

    # 获取根日志记录器
    logger = logging.getLogger()
    # 清理旧的 handler，防止重复写入或配置不生效
    if logger.hasHandlers():
        logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler() # 同时输出到控制台
        ]
    )
    for img_id in tqdm(overlap_images.keys(), total=len(overlap_images), desc="Analyzing Results"):
        img_id = int(img_id)
        cam_dict = camerasInfo[img_id]
        img_name = cam_dict['img_name'].split('/')[-1]
        width = int(cam_dict['width'])
        height = int(cam_dict['height'])
        img = cv2.imread(os.path.join(images_path, cam_dict['img_name']+'.jpg'), 0)
        
        ## 提取当前视角lines
        lines = parse_line_segments(lsd_lines_path, img_id+1, width, height).reshape(-1, 2, 2)
        lines, valid = clip_line_to_boundaries(lines, (height, width), min_len=0)
        lines = lines[valid]

        reconstructed_idx = reconstructed_idx_all.get(img_id, [])
        top3000_indices = top3000_indices_all.get(str(img_id), [])

        # 有六种情况：
        # 1. matched & reconstructed
        matched_and_reconstructed = set(corres_idx_all).intersection(set(reconstructed_idx))
        # 2. not matched & reconstructed
        not_matched_and_reconstructed = set(reconstructed_idx).difference(set(corres_idx_all))
        # 3. matched & only in top3000
        matched_and_only_top3000 = set(corres_idx_all).intersection(set(top3000_indices)).difference(set(reconstructed_idx))
        # 4. matched & not in top3000
        matched_and_not_top3000 = set(corres_idx_all).difference(set(top3000_indices))
        # 5. not matched & only in top3000
        not_matched_and_only_top3000 = set(top3000_indices).difference(set(corres_idx_all)).difference(set(reconstructed_idx))
        # 6. not matched & not in top3000
        not_matched_and_not_top3000 = set(range(len(lines))).difference(set(top3000_indices)).difference(set(corres_idx_all))

        # 可视化函数
        def visualize_line_categories(image, lines, categories, output_path, filename):
            colors = {
            "matched_and_reconstructed": (0, 255, 0),  # Green
            "not_matched_and_reconstructed": (255, 0, 0),  # Red
            "matched_and_only_top3000": (0, 0, 255),  # Blue
            "matched_and_not_top3000": (255, 255, 0),  # Cyan
            "not_matched_and_only_top3000": (255, 0, 255),  # Magenta
            "not_matched_and_not_top3000": (128, 128, 128),  # Gray
            }
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for category, indices in categories.items():
                for idx in indices:
                    line = lines[idx]
                    # 提取起点和终点，并强制转换为 int 类型的 tuple
                    # line[0] 是起点 [x1, y1], line[1] 是终点 [x2, y2]
                    pt1 = (int(line[0][1]), int(line[0][0]))
                    pt2 = (int(line[1][1]), int(line[1][0]))
                    cv2.line(vis_image, pt1, pt2, colors[category], 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(output_path, filename), vis_image)

        # 调用可视化函数
        categories = {
            "matched_and_reconstructed": matched_and_reconstructed,
            "not_matched_and_reconstructed": not_matched_and_reconstructed,
            "matched_and_only_top3000": matched_and_only_top3000,
            "matched_and_not_top3000": matched_and_not_top3000,
            "not_matched_and_only_top3000": not_matched_and_only_top3000,
            "not_matched_and_not_top3000": not_matched_and_not_top3000,
        }
        visualize_line_categories(img, lines, categories, output_path, f"{os.path.splitext(img_name)[0]}_line_categories.png")

        # 打印各情况的线段数量
        logging.info(f"  Matched & Reconstructed: {len(matched_and_reconstructed)}")
        logging.info(f"  Not Matched & Reconstructed: {len(not_matched_and_reconstructed)}")
        logging.info(f"  Matched & Only in Top3000: {len(matched_and_only_top3000)}")
        logging.info(f"  Matched & Not in Top3000: {len(matched_and_not_top3000)}")
        logging.info(f"  Not Matched & Only in Top3000: {len(not_matched_and_only_top3000)}")
        logging.info(f"  Not Matched & Not in Top3000: {len(not_matched_and_not_top3000)}")
        
        '''
        # 取交集
        matched_and_reconstructed = set(corres_idx_all).intersection(set(reconstructed_idx))
        print(f"Image {img_name}: Matched & Reconstructed lines: {len(matched_and_reconstructed)} / {len(reconstructed_idx)} reconstructed lines.")
        viz_lines2D2(img, lines[list(matched_and_reconstructed)], output_path, f"{os.path.splitext(img_name)[0]}_matched_and_recon")

        # 取重建但未被匹配到的线段
        unmatched_reconstructed = set(reconstructed_idx).difference(set(corres_idx_all))
        print(f"Image {img_name}: Reconstructed but unmatched lines: {len(unmatched_reconstructed)}")
        viz_lines2D2(img, lines[list(unmatched_reconstructed)], output_path, f"{os.path.splitext(img_name)[0]}_unmatched_recon")

        # 
        matched_and_top3000k = set(corres_idx_all).intersection(set(reconstructed_idx))
        print(f"Image {img_name}: Matched & Reconstructed lines: {len(matched_and_reconstructed)} / {len(reconstructed_idx)} reconstructed lines.")
        viz_lines2D2(img, lines[list(matched_and_reconstructed)], output_path, f"{os.path.splitext(img_name)[0]}_matched_and_recon")
        
        # Visualize the distribution of the number of matches for unmatched reconstructed lines
        unmatched_counts = [index_counts[idx] for idx in unmatched_reconstructed if idx in index_counts]
        plt.figure()
        plt.hist(unmatched_counts, bins=range(1, max(unmatched_counts) + 2), align='left', rwidth=0.8)
        plt.title(f"Match Count Distribution for Unmatched Reconstructed Lines ({img_name})")
        plt.xlabel("Number of Matches")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_path, f"{os.path.splitext(img_name)[0]}_unmatched_recon_hist.png"))
        plt.close()    
        '''

if __name__ == "__main__":
    ####################################### 参数 #######################################
    workspace = r"/home/rylynn/Pictures/datasets_3Dline/MatrixCity/block_B/" #/home/rylynn/Pictures/LinesDetection_Workspace/datasets/Dublin/block2/
    overlap_percentile = 30       # 自动阈值分位数,可以简单理解为取前%为重叠航片

    ####################################### 路径 #######################################
    sparse_model_path = os.path.join(workspace, 'sparse')
    output_path = os.path.join(workspace, 'intermediate_results_0112')
    line3dpp_path = os.path.join(output_path, 'lsd_lines_len3000')


    # 0. 数据准备：读取稀疏模型
    camerasInfo, points_in_images = load_sparse_model(sparse_model_path, image_scale=1)
    print(f"[INFO] Loaded {len(camerasInfo)} images.")

    # 1. 读取Line3D++的重建的重建结果，记录各相片实际参与重建的线段结果，记录各相片实际参与重建的线段
    _, _, lines2d_in_cam = parse_lines3dpp(line3dpp_path)
    lines2d_reconstructed = {}
    for img_id, lines2d in lines2d_in_cam.items():
        lines2d_reconstructed[img_id] = list(lines2d.keys())
    # 读取top3000线段的索引
    with open(os.path.join(line3dpp_path, 'top3000_indices.json'), 'r') as f:
        top3000_indices_all = json.load(f)
    reconstructed_idx_all = {}
    for img_id, top3000_indices in top3000_indices_all.items():
        # line3dpp中的image_id从1开始，而camerasInfo中的img_id从0开始
        line_indices = lines2d_reconstructed[int(img_id)+1]
        reconstructed_idx = [top3000_indices[idx] for idx in line_indices]
        reconstructed_idx_all[int(img_id)] = reconstructed_idx
    print(f"[INFO] Parsed reconstructed lines for {len(reconstructed_idx_all)} images.")


    # 2. 所有相片寻找邻近视角
    overlap_images = select_nearby_views(camerasInfo, points_in_images, overlap_percentile)

    # 3. 根据邻近视角，寻找有4次及以上匹配的线段
    index_counts, corres_idx_all = project_and_establish_correspondences(
        camerasInfo, overlap_images,reconstructed_idx_all, top3000_indices_all,
        lsd_lines_path=os.path.join(output_path, 'lsd_lines_all'),
        images_path=os.path.join(workspace, 'images'),
        depth_path=os.path.join(workspace, 'depth_maps'),
        output_path=output_path,
        neighbor_thred=8
    )
    save_dict_to_json(corres_idx_all, os.path.join(output_path, 'corres1_idx_all.json'))

    # 4. 对比步骤1和步骤3的结果，分析步骤3是否能较好地预测哪些线段会被Line3D++重建
    '''
    analyze_results(index_counts, corres_idx_all, reconstructed_idx_all, top3000_indices_all, 
                    images_path=os.path.join(workspace, 'images'), 
                    lsd_lines_path=os.path.join(output_path, 'lsd_lines_all'), 
                    output_path=output_path)
    '''