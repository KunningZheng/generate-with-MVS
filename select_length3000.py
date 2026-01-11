import os
import json
import numpy as np
from tqdm import tqdm
import random
import networkx as nx
import matplotlib.pyplot as plt
import cv2

from datasets.dataset_reader import (
    load_sparse_model,
    match_pair,
    read_depth
)
from datasets.line3dpp_loader import parse_line_segments, parse_lines3dpp, save_segments_l3dpp
from utils.visualize import viz_lines2D2

if __name__ == "__main__":
    ####################################### 参数 #######################################
    workspace = r"/home/rylynn/Pictures/LinesDetection_Workspace/datasets/Dublin/block2/"

    ####################################### 路径 #######################################
    sparse_model_path = os.path.join(workspace, 'sparse')
    images_path = os.path.join(workspace, 'images')
    output_path = os.path.join(workspace, 'intermediate_results_0104')
    lsd_lines_path = os.path.join(output_path, 'lsd_lines_all')
    len3000_path = os.path.join(output_path, 'lsd_lines_len3000')
    os.makedirs(len3000_path, exist_ok=True)
    viz_path = os.path.join(len3000_path, 'viz_lsd_len3000')
    os.makedirs(viz_path, exist_ok=True)
    

    # 0. 数据准备：读取稀疏模型，读取LSD检测所有线段
    camerasInfo, points_in_images = load_sparse_model(sparse_model_path, image_scale=1)
    print(f"[INFO] Loaded {len(camerasInfo)} images.")

    # 1. 取长度前3000的线段存储为Line3D++的输入
    topk_indices_all = {}
    for img_id, cam_dict in tqdm(enumerate(camerasInfo), total=len(camerasInfo), desc="Processing lines"):
        cam_dict = camerasInfo[img_id]
        width = int(cam_dict['width'])
        height = int(cam_dict['height'])
        img_name = cam_dict['img_name'].split('/')[-1]

        lines = parse_line_segments(lsd_lines_path, img_id+1, width, height)[:, [1,0,3,2]]


        # 取长度前3000的线段
        lines_lengths = ((lines[:, 0]-lines[:, 2])**2 + (lines[:, 1]-lines[:, 3])**2)**0.5

        # 对长度进行降序排序，获取对应的原始索引
        # argsort 返回的是：原本在哪个位置的元素现在排在这里
        sorted_indices = np.argsort(-lines_lengths)
        # 取前 3000 个索引
        top_k = min(3000, lines.shape[0])  # 防止 lines 总数不足 3000
        topk_indices = sorted_indices[:top_k]
        # 根据索引提取对应的线段
        lines_topk = lines[topk_indices]
        topk_indices_all[img_id] = topk_indices.tolist()
 
        #save_segments_l3dpp(lines_topk, len3000_path, img_id+1, width, height)
        img = cv2.imread(os.path.join(images_path, cam_dict['img_name']+'.jpg'), 0)
        viz_lines2D2(img, lines_topk[:, [1,0,3,2]].reshape(-1,2,2), viz_path, f"{os.path.splitext(img_name)[0]}")
    
    # 保存top3000线段的索引
    #with open(os.path.join(len3000_path, 'top3000_indices.json'), 'w') as f:
        #json.dump(topk_indices_all, f)

