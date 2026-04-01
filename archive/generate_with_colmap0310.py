import os
import sys
import json
import shutil
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import cv2
from functools import reduce
import matplotlib.pyplot as plt
import h5py

from datasets.dataset_reader import load_sparse_model, match_pair, read_depth, find_common_points, compute_bounding_box
from transformation.views_transform_fang import views_transform_lsd
from utils.visualize import viz_lines2D2
from utils.line_tools import af_df_producer

from deeplsd.geometry.viz_2d import get_flow_vis




def resize_image(img, scale):
    '''
    缩小图片尺寸
    '''
    # 计算新尺寸
    new_height = img.shape[0] // scale  # 如果前后都是int，则//输出int
    new_width = img.shape[1] // scale
    # 缩小图像
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


def main():
    ####################################### 需要手动改变的参数 #######################################
    workspace = r"/home/rylynn/Pictures/LinesDetection_Workspace/datasets/Dublin/block2/"
    # 并行数量
    n_jobs = 1
    # 单应性变换的数量
    num_H = 1
    # 是否进行random_contrast
    rdm_contrast = False
    # 图像缩小
    image_scale = 1

    ####################################### 预处理 #######################################
    sparse_model_path = os.path.join(workspace, 'sparse')
    images_path = os.path.join(workspace, 'images')
    depth_path = os.path.join(workspace, 'depth_maps')
    output_path = os.path.join(workspace, 'intermediate_results')
    gt_hdf5_path = os.path.join(workspace, 'gt', 'hdf5')
    gt_img_path = os.path.join(workspace, "gt", "images")
    os.makedirs(output_path,exist_ok=True)
    os.makedirs(gt_hdf5_path, exist_ok=True)
    os.makedirs(gt_img_path, exist_ok=True)

    # 读取sparse model
    camerasInfo, _ = load_sparse_model(sparse_model_path, image_scale)
    # 释放point的键值对
    for cam_dict in camerasInfo:
        if "points3D_ids" in cam_dict:  # 检查键是否存在，避免 KeyError
            del cam_dict["points3D_ids"]
        if "points3D_to_xys" in cam_dict:
            del cam_dict["points3D_to_xys"]

    # 读取output_path中near_image_ids_**.json
    json_files = [f for f in os.listdir(output_path) if f.startswith("near_image_ids_") and f.endswith(".json")]
    if len(json_files) == 0:
        print("[ERROR] No near_image_ids_**.json file found in output_path.")
        return
    # 假设只处理第一个找到的文件
    json_file_path = os.path.join(output_path, json_files[0])
    with open(json_file_path, "r") as f:
        json_str = f.read()
        near_image_ids = json.loads(json_str)
    

    ####################################### 伪真值生成 #######################################
    # 当前视角加载
    for img_id, nimg_ids in near_image_ids.items():
        img_id = int(img_id)
        cam_dict = camerasInfo[img_id]
        img_name = cam_dict['img_name'].split('/')[-1]
        print(img_name+" start")
        
        img = cv2.imread(os.path.join(images_path, cam_dict['img_name']+'.jpg'), 0)
        img = resize_image(img, image_scale)
        depth = read_depth(os.path.join(depth_path, cam_dict['img_name']+'.jpg.geometric.bin'))
        depth = depth / image_scale

        dfs, angles, closests, raster_lines = [], [], [], []
        # 新视角加载
        for nimg_id in tqdm(nimg_ids):
            ncam_dict = camerasInfo[nimg_id]
            nimg_name = ncam_dict['img_name'].split('/')[-1]
            nimg = cv2.imread(os.path.join(images_path, ncam_dict['img_name']+'.jpg'), 0)
            nimg = resize_image(nimg, image_scale)
            ndepth = read_depth(os.path.join(depth_path, ncam_dict['img_name']+'.jpg.geometric.bin'))
            ndepth = ndepth / image_scale

            # 视角转换
            nlines = views_transform_lsd(img, depth, cam_dict, nimg, ndepth, ncam_dict)
            # test: 可视化
            #viz_lines2D2(img, nlines, output_path, str(nimg_id))

            # 获取af和df
            df, angle, closest, raster_line = af_df_producer(nlines, img)
            dfs.append(df)
            angles.append(angle)
            closests.append(closest)
            raster_lines.append(raster_line)

        dfs, angles, closests = np.stack(dfs), np.stack(angles), np.stack(closests)
        # gt_df:在不同视角的平均值中取最小值
        gt_df = np.nanmin(dfs, axis=0)      
        min_index = np.nanargmin(dfs, axis=0)
        # gt_angles, gt_closets:选取gt_df最小值处的角度值（配套）
        rows, cols = np.indices(min_index.shape)
        gt_angle = angles[min_index, rows, cols]  # 创建一个网格，其中 rows 和 cols 分别表示 min_index 中每个元素的行和列索引
        gt_closest = closests[min_index, rows, cols]
        # raster_lines取并集
        gt_raster_lines = reduce(np.logical_or, raster_lines).astype(np.uint8)  # reduce：逐个应用
        gt_raster_lines = cv2.dilate(gt_raster_lines, np.ones((21, 21), dtype=np.uint8))
        gt_bg_mask = (1 - gt_raster_lines).astype(float)

        # Visualize
        plt.imsave(os.path.join(output_path, img_name+ '_df.jpg'), gt_df, cmap='viridis_r')
        angle_field= get_flow_vis(gt_df, gt_angle)
        plt.imsave(os.path.join(output_path, img_name+ '_angle.jpg'), angle_field)
        plt.imsave(os.path.join(output_path, img_name+ '_raster_lines.jpg'), gt_raster_lines, cmap='binary')
        plt.imsave(os.path.join(output_path, img_name+ '_bg_mask.jpg'), gt_bg_mask, cmap='binary')

        # 存储真值
        # IMAGE
        src_path = os.path.join(images_path, cam_dict['img_name']+'.jpg')
        dst_path = os.path.join(gt_img_path, img_name+'.jpg')
        shutil.copy(src_path, dst_path)
        # HDF5
        out_path = os.path.join(gt_hdf5_path, img_name) + '.hdf5'
        with h5py.File(out_path, "w") as f:
            f.create_dataset("df", data=gt_df.flatten())
            f.create_dataset("line_level", data=gt_angle.flatten())
            f.create_dataset("closest", data=gt_closest.flatten())
            f.create_dataset("bg_mask", data=gt_bg_mask.flatten())   

        print(img_name+" finish")
        




if __name__ == "__main__":
    main()