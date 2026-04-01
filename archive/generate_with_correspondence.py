import os
import json
import cv2
import h5py


from pytlsd import lsd
from deeplsd.geometry.line_utils import clip_line_to_boundaries
from utils.tile import tile_and_save_images, load_tiled_raster_lines, tiled_df_to_raster_lines
from utils.homography_adaptation_df import export_ha
from utils.config import get_config, PathManager
from datasets.dataset_reader import load_sparse_model, read_depth
from transformation.views_transform_fang import views_transform_lsd
from utils.line_tools import establish_line_correspondences
from utils.visualize import viz_lines2D2

def project_and_establish_correspondences(near_image_ids, camerasInfo, images_path, depth_path, output_path):
    output_path = os.path.join(output_path, 'correspondence')
    ## 逐张影像、逐邻近视角循环
    for img_id, nimg_ids in near_image_ids.items():
        if len(nimg_ids) < 11:
            print(f"[WARNING] Image ID {img_id} has less than 8 neighbors, skipping.")
            continue
        img_id = int(img_id)
        cam_dict = camerasInfo[img_id]
        img_name = cam_dict['img_name'].split('/')[-1]
        img = cv2.imread(os.path.join(images_path, cam_dict['img_name']+'.jpg'), 0)
        depth = read_depth(os.path.join(depth_path, cam_dict['img_name']+'.jpg.geometric.bin'))
        
        # 提取当前视角lines
        lines = lsd(img)[:, [1, 0, 3, 2]].reshape(-1, 2, 2)
        lines, valid = clip_line_to_boundaries(lines, img.shape, min_len=0)
        lines = lines[valid]

        ## TEST: 可视化当前视角前3000长度的线段        
        # 按线段长度排序，选择前3000条最长的线段
        line_lengths = ((lines[:, 1] - lines[:, 0]) ** 2).sum(axis=1) ** 0.5
        top_indices = line_lengths.argsort()[-3000:][::-1]
        top_lines = lines[top_indices]

        # 可视化前3000条线段
        viz_lines2D2(img, lines, output_path, f"{os.path.splitext(img_name)[0]}")
        viz_lines2D2(img, top_lines, output_path, f"{os.path.splitext(img_name)[0]}_top1000")
        
        valid_idx_all = []
        for nimg_id in nimg_ids:
            ## 跳过当前视角
            if int(nimg_id) == int(img_id):
                continue
            ncam_dict = camerasInfo[nimg_id]
            nimg_name = ncam_dict['img_name'].split('/')[-1]
            nimg = cv2.imread(os.path.join(images_path, ncam_dict['img_name']+'.jpg'), 0)
            ndepth = read_depth(os.path.join(depth_path, ncam_dict['img_name']+'.jpg.geometric.bin'))

            ## Step1: 投影线段，裁剪到当前视角范围内
            nlines = views_transform_lsd(img, depth, cam_dict, nimg, ndepth, ncam_dict)
            matches = establish_line_correspondences(lines, nlines, t_angle=1.0, 
                                   t_dist=1.0, 
                                   t_overlap=0.95)

            valid_idx = set()
            for _, l_idx, _ in matches:
                valid_idx.add(l_idx)
            valid_idx = list(valid_idx)
            viz_lines2D2(img, lines[valid_idx], output_path, f"{os.path.splitext(img_name)[0]}_matched_{nimg_id}")
            valid_idx_all.append(valid_idx)
        
            '''
            # 保存matches
            matches_path = os.path.join(output_path, 'line_correspondences')
            os.makedirs(matches_path, exist_ok=True)
            matches_file = os.path.join(matches_path, f"{os.path.splitext(img_name)[0]}_to_{os.path.splitext(nimg_name)[0]}_matches.json")
            with open(matches_file, 'w') as f:
                json.dump([{'current_line_idx': l_idx, 'projected_line_idx': n_idx, 'score': score} 
                           for n_idx, l_idx, score in matches], f, indent=4)
            '''
        # 保留出现4次及以上的线段
        from collections import Counter
        all_indices = [idx for sublist in valid_idx_all for idx in sublist]
        index_counts = Counter(all_indices)
        valid_idx_all = [idx for idx, count in index_counts.items() if count >= 2]
        print(f"Image {img_name}: Total matched lines from neighbors: {len(valid_idx_all)} / {len(lines)}")
        viz_lines2D2(img, lines[list(valid_idx_all)], output_path, f"{os.path.splitext(img_name)[0]}_matched")




def main():
    ####################### Stage0: Preparation #######################
    ## Load configuration
    conf = get_config()
    pth_m = PathManager(conf['workspace'])
    pth_m.create_paths()

    # 读取sparse model
    camerasInfo, _ = load_sparse_model(pth_m.sparse_model_path, image_scale=1)
    for cam_dict in camerasInfo:
        if "points3D_ids" in cam_dict: del cam_dict["points3D_ids"]
        if "points3D_to_xys" in cam_dict: del cam_dict["points3D_to_xys"]
    
    # 读取 near_image_ids
    json_files = [f for f in os.listdir(pth_m.output_path) if f.startswith("near_image_ids_") and f.endswith(".json")]
    if len(json_files) == 0:
        print("[ERROR] No near_image_ids_**.json file found.")
        return
    with open(os.path.join(pth_m.output_path, json_files[0]), "r") as f:
        near_image_ids = json.load(f)


    # 记录不重复航片的文件名
    nonoverlap_img_names = []
    for img_id, _ in near_image_ids.items():
        cam_dict = camerasInfo[int(img_id)]
        nonoverlap_img_names.append(cam_dict['img_name']+'.jpg')

    project_and_establish_correspondences(near_image_ids, camerasInfo, 
                                         pth_m.images_path, pth_m.depth_path, pth_m.gt_path)
    


if __name__ == "__main__":
    main()