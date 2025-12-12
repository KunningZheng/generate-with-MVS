import os
import json
import cv2
import h5py

from utils.tile import tile_and_save_images, load_tiled_raster_lines, tiled_df_to_raster_lines
from utils.homography_adaptation_df import export_ha
from utils.config import get_config, PathManager
from datasets.dataset_reader import load_sparse_model, read_depth
from transformation.views_transform_fang import views_transform_lsd
from utils.line_tools import select_lines, af_df_producer
from utils.visualize import viz_lines2D2

####################### Stage1: Tile and Avarage #######################
def tile_and_average(images_path, nonoverlap_img_names, output_folder, 
                     patch_size=2048, overlap=0,
                     num_H=100, random_contrast=True, n_jobs=1):
    '''
        - images_path: 存储原始影像的路径
        - nonoverlap_img_names: 不重复航片的文件名列表
        - output_folder: 存储分块影像和Anchor Map的路径
    '''
    ## Step0: 创建存储分块影像的文件夹
    images_patches_path = os.path.join(output_folder, 'images_patches')
    os.makedirs(images_patches_path, exist_ok=True)
    hdf5_path = os.path.join(output_folder, 'anchor_maps')
    os.makedirs(hdf5_path, exist_ok=True)

    ## Step1: 分块为2048*2048，存储分块影像
    print("Tiling images into patches...")
    tile_and_save_images(images_path, nonoverlap_img_names, images_patches_path, 
                         patch_size, overlap)

    ## Step2: 逐分块100次H变换+平均，获取Anchor Map
    print("Generating Anchor Maps with Homography Adaptation...")
    
    images_patches_list = os.path.join(images_patches_path, 'images_patches_list.txt')
    export_ha(images_patches_list, hdf5_path, 
              num_H, random_contrast, n_jobs)
    
    print("Anchor Maps generation completed.")


####################### Stage2: Obtain Supplentary Information from Neighbors #######################
def obtain_from_neighbors(near_image_ids, camerasInfo, images_path, depth_path, output_path,
                          patch_size, overlap,
                          retain_ratio=0.3):
    ## 逐张影像、逐邻近视角循环
    for img_id, nimg_ids in near_image_ids.items():
        img_id = int(img_id)
        cam_dict = camerasInfo[img_id]
        img_name = cam_dict['img_name'].split('/')[-1]
        img = cv2.imread(os.path.join(images_path, cam_dict['img_name']+'.jpg'), 0)
        depth = read_depth(os.path.join(depth_path, cam_dict['img_name']+'.jpg.geometric.bin'))

        ## Step0: 加载当前视角的Achor Map中的Raster_lines，合并为一张大图
        raster_lines = tiled_df_to_raster_lines(img, img_name, output_path, patch_size, overlap)
        ## TEST: 可视化raster_lines
        viz_path = os.path.join(output_path, 'viz_raster_lines')
        os.makedirs(viz_path, exist_ok=True)
        cv2.imwrite(os.path.join(viz_path, f"{os.path.splitext(img_name)[0]}_raster_lines.jpg"), raster_lines*255)
        
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
            
            ## Step2: 判定是否保留线段（利用Stage1获取的raster_lines）
            # Check if 30% of the line falls within the raster_lines region
            retained_nlines = select_lines(img, nlines, raster_lines, retain_ratio)
            ## TEST: 可视化retained_lines
            viz_path = os.path.join(output_path, 'viz_retained_lines')
            os.makedirs(viz_path, exist_ok=True)
            viz_lines2D2(img, retained_nlines, viz_path, f"{os.path.splitext(img_name)[0]}_from_{nimg_id}")

            ## Step3: 获取Proposal Map
            ndf, nangle, nclosest, nraster_lines = af_df_producer(retained_nlines, img)
            nbg_mask = 1 - nraster_lines

            ## Step4: 存储结果
            proposal_map_path = os.path.join(output_path, 'proposal_maps', 
                                             f"{os.path.splitext(img_name)[0]}_from_{nimg_id}.hdf5")
            os.makedirs(os.path.dirname(proposal_map_path), exist_ok=True)
            with h5py.File(proposal_map_path, 'w') as hdf5_file:
                hdf5_file.create_dataset("df", data=ndf.flatten())
                hdf5_file.create_dataset("line_level", data=nangle.flatten())
                hdf5_file.create_dataset("closest", data=nclosest.flatten())
                hdf5_file.create_dataset("bg_mask", data=nbg_mask.flatten())


####################### Stage3: Fuse Anchor Map and Proposal Map #######################
def fuse_anchor_and_proposal():
    ## 逐Proposal Map循环
    ## Step1: 读取Proposal Map，分块

    ## Step2: 读取对应的分块Anchor Map

    ## Step3: 合并两张Map（取极值的方法）
    return

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

    ####################### Stage1: Tile and Avarage #######################
    print("### Stage1: Tile and Avarage ###")
    tile_and_average(pth_m.images_path, nonoverlap_img_names, pth_m.gt_path, 
                     conf['patch_size'], conf['overlap'],
                     conf['num_H'], conf['random_contrast'], conf['n_jobs'])

    ####################### Stage2: Obtain Supplentary Information from Neighbors #######################
    obtain_from_neighbors(near_image_ids, camerasInfo, pth_m.images_path, pth_m.depth_path, pth_m.gt_path,
                          conf['patch_size'], conf['overlap'], conf['retain_ratio'])

    ####################### Stage3: Fuse Anchor Map and Proposal Map #######################
    fuse_anchor_and_proposal()


if __name__ == "__main__":
    main()