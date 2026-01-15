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
from transformation.views_transform_fang import views_transform_reverse, grid_reprojection_
from utils.line_tools import establish_line_correspondences_reverse
from utils.visualize import viz_lines2D2

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

def find_vis_in_neighbor(camerasInfo, overlap_images, depth_path, output_path, vis_thred=95):
    
    images_vis_in_neighbor = []
    for img_id, nimg_ids in tqdm(overlap_images.items(), total=len(overlap_images), desc="Correspondence Establishment"):
        img_id = int(img_id)
        cam_dict = camerasInfo[img_id]
        img_name = cam_dict['img_name'].split('/')[-1]
        width = int(cam_dict['width'])
        height = int(cam_dict['height'])
        # 确定是.png还是.jpg
        if os.path.exists(os.path.join(depth_path, cam_dict['img_name']+'.png.geometric.bin')):
            depth = read_depth(os.path.join(depth_path, cam_dict['img_name']+'.png.geometric.bin'))
        else:
            depth = read_depth(os.path.join(depth_path, cam_dict['img_name']+'.jpg.geometric.bin'))

        overlap_area_all = np.zeros((height, width))
        for nimg_id in nimg_ids:
            # 跳过当前视角
            if int(nimg_id) == int(img_id):
                continue
            ncam_dict = camerasInfo[nimg_id]
            nimg_name = ncam_dict['img_name'].split('/')[-1]
            nwidth = int(ncam_dict['width'])
            nheight = int(ncam_dict['height'])

            # 重叠区域
            overlap_mask = grid_reprojection_([nheight, nwidth], ncam_dict, [height, width], depth, cam_dict)
            # 添加到总体的重叠中
            overlap_area_all = overlap_area_all + overlap_mask
        
        plt.imshow(overlap_area_all.astype(bool), cmap='gray')
        plt.title(f"Overlap Mask: {img_name}")
        plt.axis('off')
        plt.show()
        # 存储overlap_area_all
        overlap_save_path = os.path.join(output_path, f"overlap_mask_{img_id}.npy")
        np.save(overlap_save_path, overlap_area_all.astype(bool))
        # 判别总体与其他视角的重叠区域是否超过vis_thred
        overlap_total_num = np.sum(np.sum(overlap_area_all>0))
        if overlap_total_num / (width*height) >= vis_thred:
            images_vis_in_neighbor.append(img_id)
    print(f'total {len(images_vis_in_neighbor)} images visible in neighborhood')
    


if __name__ == "__main__":
    ####################################### 参数 #######################################
    workspace = r"/home/rylynn/Pictures/LinesDetection_Workspace/datasets/Dublin/block2/" 
    overlap_percentile = 90       # 自动阈值分位数,可以简单理解为取前%为重叠航片
    visible_thredshold = 95       # 在邻近视角中超过95%区域可见，认为可以作为训练数据

    ####################################### 路径 #######################################
    sparse_model_path = os.path.join(workspace, 'sparse')
    output_path = os.path.join(workspace, 'intermediate_results_0115')
    line3dpp_path = os.path.join(output_path, 'lsd_lines_len3000')
    depth_path=os.path.join(workspace, 'depth_maps')


    # 0. 数据准备：读取稀疏模型
    camerasInfo, points_in_images = load_sparse_model(sparse_model_path, image_scale=1)
    print(f"[INFO] Loaded {len(camerasInfo)} images.")

    # 1. 所有相片寻找邻近视角
    overlap_images = select_nearby_views(camerasInfo, points_in_images, overlap_percentile)

    # 2. 筛选在邻近视角中超过95%的区域可见的相片
    images_vis_in_neighbor = find_vis_in_neighbor(camerasInfo, overlap_images, 
                                                  depth_path, output_path, visible_thredshold)
    