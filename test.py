import os
import json
import numpy as np
from tqdm import tqdm
import random
import networkx as nx
import matplotlib.pyplot as plt

from datasets.dataset_reader import (
    load_sparse_model,
    match_pair,
    find_common_points,
    compute_bounding_box
)


def main():
    ####################################### 参数 #######################################
    sparse_model_path = r"/home/rylynn/Pictures/Clustering_Workspace/Shanghai_Region5/Colmap/sparse"
    line2d_path = r"/home/rylynn/Pictures/datasets_3Dline/Shanghai_Region5/sliding1024_md/Line3D++/L3D++_data/"
    camerasInfo, _ = load_sparse_model(sparse_model_path, image_scale=1)
    for cam_id, cam_dict in enumerate(camerasInfo):
        width = int(cam_dict['width'])
        height = int(cam_dict['height'])
        img_name = cam_dict['img_name']
        line2d_filename1 = f'segments_L3D++_{img_name}_{width}x{height}.bin'
        line2d_filename2 = f'segments_L3D++_{cam_id}_{width}x{height}.bin'
        old_path = os.path.join(line2d_path, line2d_filename1)
        new_path = os.path.join(line2d_path, line2d_filename2)
        os.rename(old_path, new_path)
        print(f"{img_name}更名成功")


if __name__ == "__main__":
    main()