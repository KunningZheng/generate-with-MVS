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

from datasets.dataset_reader import load_sparse_model, match_pair, read_depth, find_common_points, compute_bounding_box
from transformation.views_transform_fang import views_transform_lsd, views_transform, grid_reprojection
from utils.visualize import viz_lines2D2
from utils.line_tools import af_df_producer

from deeplsd.geometry.viz_2d import get_flow_vis
from deeplsd.datasets.utils.homographies import sample_homography, warp_lines
from utils.line_tools import clip_lines_to_image


homography_params = {
    'translation': True,
    'rotation': True,
    'scaling': True,
    'perspective': True,
    'scaling_amplitude': 0.1,           # 减小
    'perspective_amplitude_x': 0.05,    # 减小
    'perspective_amplitude_y': 0.05,    # 减小
    'patch_ratio': 0.95,                # 增大
    'max_angle': 1.57,
    'allow_artifacts': True
}


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


def save_homography_lines(out_path, H_list, lines_list):
    """
    H_list: list of (3,3)
    lines_list: list of (N_i, 2, 2)  每次 homography 得到的线段
    """

    num_H = len(H_list)
    max_lines = max([len(l) for l in lines_list])

    with h5py.File(out_path, 'w') as f:
        
        # 保存所有 H
        f.create_dataset("homographies", data=np.stack(H_list)) # shape = (100,3,3)
        
        # 保存每次的线段数量
        line_counts = np.array([len(l) for l in lines_list], dtype=np.int32)
        f.create_dataset("line_counts", data=line_counts)       # shape = (100,)
        
        # 创建最大空间的数据集（ragged array via padding）
        dset = f.create_dataset(
            "lines",
            shape=(num_H, max_lines, 2, 2),
            dtype=np.float32,
            fillvalue=np.nan
        )
        
        # 写入每一次 H 的线段
        for i, lines in enumerate(lines_list):
            dset[i, :len(lines)] = lines.astype(np.float32)


def Homography_adaptation(cam_id, camerasInfo, homography_outpath, images_path, output_path, num_H):
    cam_dict = camerasInfo[cam_id]
    img_name = cam_dict['img_name'].split('/')[-1]

    # 如果hdf5文件已存在，则跳过
    #if os.path.exists(os.path.join(homography_outpath, img_name+'_homography.hdf5')):
        #return

    img = cv2.imread(os.path.join(images_path, cam_dict['img_name']+'.jpg'), 0)
    h, w = img.shape[:2]
    size = (w, h)        

    H_list = []
    lines_list = []
    # Loop through all the homographies
    for i in range(num_H):
        # Generate a random homography
        if i == 0:
            H = np.eye(3)
        else:
            H = sample_homography(img.shape, **homography_params)
        H_inv = np.linalg.inv(H)
        H_list.append(H)
        
        # Warp the image
        warped_img = cv2.warpPerspective(img, H, size,
                                        borderMode=cv2.BORDER_REPLICATE)
        
        # Regress the DF on the warped image
        warped_lines = lsd(warped_img)[:, [1, 0, 3, 2]].reshape(-1, 2, 2)
        
        # Warp the lines back
        lines = warp_lines(warped_lines, H_inv)
        # Clip lines to image（LSD算法的线段延长特性导致线段端点超出范围）
        lines = clip_lines_to_image(lines, h, w)
        lines_list.append(lines)
        ## test
        #viz_lines2D2(img, lines, os.path.join(output_path, 'ctest'), str(i))
    save_homography_lines(os.path.join(homography_outpath, img_name+'_homography.hdf5'), H_list, lines_list)    


def Aggregate_by_averaging(cam_dict, img, depth, ncam_dict, nimg, ndepth, nlines_list, H_list, output_path, border_margin, min_counts):
    '''
    输出和img尺寸一样
    '''
    h, w = img.shape[:2]

    dfs, angles, closests, counts = [], [], [], []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                (border_margin * 2, border_margin * 2))
    pix_loc = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='ij'),
                        axis=-1)
    raster_lines = np.zeros_like(img)
    i = 0
    for H, nlines in zip(tqdm(H_list, desc="Views_transformer"), nlines_list):
        # 视角转换
        lines = views_transform(img, nlines, cam_dict, nimg, ndepth, ncam_dict)
        # test: 可视化
        nimg_name = ncam_dict['img_name'].split('/')[-1]
        #viz_lines2D2(img, lines, output_path, nimg_name)

        # Get the DF and angles
        num_lines = len(lines)
        cuda_lines = torch.from_numpy(lines[:, :, [1, 0]].astype(np.float32))
        cuda_lines = cuda_lines.reshape(-1, 4)[None].cuda()
        offset = afm(
            cuda_lines,
            torch.IntTensor([[0, num_lines, h, w]]).cuda(), h, w)[0]
        offset = offset[0].permute(1, 2, 0).cpu().numpy()[:, :, [1, 0]]
        closest = pix_loc + offset
        df = np.linalg.norm(offset, axis=-1)
        angle = np.mod(np.arctan2(
            offset[:, :, 0], offset[:, :, 1]) + np.pi / 2, np.pi)
        
        dfs.append(df)
        angles.append(angle)
        closests.append(closest)
        
        # Compute the valid pixels
        count = grid_reprojection(nimg, ndepth, ncam_dict, img, depth, cam_dict, H)
        i +=1
        count = count.astype('uint8')  # 转成uint8，才能进行cv2.erode
        count = cv2.erode(count, kernel)
        ## test:可视化并存储count
        #cv2.imwrite(os.path.join(output_path, 'avg', nimg_name+'_'+str(i)+'_count.jpg'), (count / count.max() * 255).astype(np.uint8))
        counts.append(count)
        raster_lines += (df < 1).astype(np.uint8) * count 
        
    # Compute the median of all DF maps, with counts as weights
    dfs, angles = np.stack(dfs), np.stack(angles)
    counts, closests = np.stack(counts), np.stack(closests)
    
    # Median of the DF
    dfs[counts == 0] = np.nan
    avg_df = np.nanmedian(dfs, axis=0)  # 如果count都为0，平均后该像素位置就会变成nan

    # Median of the closest
    closests[counts == 0] = np.nan
    avg_closest = np.nanmedian(closests, axis=0)

    # Median of the angle
    circ_bound = (np.minimum(np.pi - angles, angles)
                * counts).sum(0) / counts.sum(0) < 0.3
    angles[:, circ_bound] -= np.where(
        angles[:, circ_bound] > np.pi /2,
        np.ones_like(angles[:, circ_bound]) * np.pi,
        np.zeros_like(angles[:, circ_bound]))
    angles[counts == 0] = np.nan
    avg_angle = np.mod(np.nanmedian(angles, axis=0), np.pi)

    # Generate the background mask and a saliency score
    raster_lines = np.where(raster_lines > min_counts, np.ones_like(img),
                            np.zeros_like(img))
    raster_lines = cv2.dilate(raster_lines, np.ones((21, 21), dtype=np.uint8))
    #bg_mask = (1 - raster_lines).astype(float) 

    return avg_df, avg_angle, avg_closest, raster_lines 


def main():
    ####################################### 需要手动改变的参数 #######################################
    workspace = r"/home/rylynn/Pictures/LinesDetection_Workspace/datasets/Dublin/block2/"
    # 并行数量
    n_jobs = 8
    # 单应性变换的数量
    num_H = 20
    # 是否进行random_contrast
    rdm_contrast = False
    # 图像缩小
    image_scale = 1
    border_margin=3
    min_counts=7

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

            # 读取单应性变换得到的线段
            with h5py.File(os.path.join(homography_outpath, nimg_name+'_homography.hdf5'), 'r') as f:
                H_list = f["homographies"][:]              # (100, 3, 3)
                line_counts = f["line_counts"][:]          # (100,)
                lines_all = f["lines"][:]                  # (100, max_lines, 2, 2)
            # 恢复每次的真实线段
            nlines_list = [lines_all[i, :line_counts[i]] for i in range(num_H)]
            H_list = [H_list[i] for i in range(num_H)]
            ## 平均
            avg_df, avg_angle, avg_closest, avg_raster_lines = Aggregate_by_averaging(cam_dict, img, depth, 
                                                                                      ncam_dict, nimg, ndepth, nlines_list, H_list, 
                                                                                      output_path, border_margin, min_counts)
            # Visualize
            avg_df_masked = np.ma.masked_invalid(avg_df)  # 创建masked array，自动处理NaN
            plt.imsave(os.path.join(output_path, "avg", img_name+'_'+nimg_name+ '_df.jpg'), avg_df_masked, cmap='viridis_r')
            plt.imsave(os.path.join(output_path, "avg",img_name+'_'+nimg_name+ '_raster_lines.jpg'), avg_raster_lines, cmap='binary')       
            dfs.append(avg_df)
            angles.append(avg_angle)
            closests.append(avg_closest)
            raster_lines.append(avg_raster_lines)               

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
        os.makedirs(os.path.join(output_path, 'visualize'), exist_ok=True)
        plt.imsave(os.path.join(output_path, 'visualize', img_name+ '_df.jpg'), gt_df, cmap='viridis_r')
        angle_field= get_flow_vis(gt_df, gt_angle)
        plt.imsave(os.path.join(output_path, 'visualize', img_name+ '_angle.jpg'), angle_field)
        plt.imsave(os.path.join(output_path, 'visualize', img_name+ '_raster_lines.jpg'), gt_raster_lines, cmap='binary')
        plt.imsave(os.path.join(output_path, 'visualize', img_name+ '_bg_mask.jpg'), gt_bg_mask, cmap='binary')

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