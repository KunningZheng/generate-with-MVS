import os
import json
import numpy as np
from tqdm import tqdm
import random
import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import math

from datasets.dataset_reader import (
    load_sparse_model,
    match_pair,
    find_common_points,
    compute_bounding_box,
    read_depth
)
from transformation.views_transform_fang import grid_reprojection_


def build_overlap_graph(camerasInfo, points_in_images, match_point_num):
    """构建图：节点为图像，边表示重叠"""
    _, overlap_images = match_pair(camerasInfo, points_in_images, match_point_num=match_point_num)
    G = nx.Graph()
    for cam_dict in camerasInfo:
        G.add_node(cam_dict['id'])
    for i, matched_ids in overlap_images.items():
        for j in matched_ids:
            if i != j:
                G.add_edge(i, j)
    return G, overlap_images


def choose_nonoverlapping_images(G, camerasInfo):
    """空间有序、结果稳定地选取不重叠航片"""
    random.seed(42)  # 设定随机种子
    center_pos = np.mean([c['position'] for c in camerasInfo], axis=0)
    distances = {c['id']:np.linalg.norm(np.array(c['position']) - center_pos) for c in camerasInfo}
    sorted_nodes = sorted(distances, key=distances.get)
    def greedy_mis_with_order(G, ordered_nodes):
        """
        Greedy maximal independent set following a fixed node order.
        返回列表形式的节点 id。
        """
        S = set()
        for n in ordered_nodes:
            # 如果 n 已被排除（与 S 中某节点相邻），跳过
            if n in S:
                continue
            # 检查 S 中是否有 n 的邻居
            has_nei_in_S = False
            for nb in G[n]:
                if nb in S:
                    has_nei_in_S = True
                    break
            if not has_nei_in_S:
                S.add(n)
        return list(S)
    nonoverlap_ids = greedy_mis_with_order(G, sorted_nodes)
    return nonoverlap_ids
    


def compute_overlap_ratio(cam_dict1, cam_dict2, common_points):
    """计算重叠比例"""
    bb_area1 = compute_bounding_box(common_points[:, 0])
    bb_area2 = compute_bounding_box(common_points[:, 1])
    area1 = cam_dict1['width'] * cam_dict1['height']
    area2 = cam_dict2['width'] * cam_dict2['height']
    return bb_area1 / area1, bb_area2 / area2


def visualize_camera_distribution(camerasInfo, nonoverlap_ids, near_image_ids, output_path):
    """绘制不重叠航片及其邻近航片分布"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    positions = np.array([cam['position'] for cam in camerasInfo])
    xs, ys = positions[:, 0], positions[:, 1]

    # 所有航片：灰色
    ax.scatter(xs, ys, c='lightgray', s=15, label='All Images')

    # 不重叠航片：蓝色
    ax.scatter(xs[nonoverlap_ids], ys[nonoverlap_ids], c='blue', s=35, label='Non-overlapping')

    # 邻近航片关系线：浅红色
    for key, neighbors in near_image_ids.items():
        p1 = np.array(camerasInfo[key]['position'])[:2]
        for n in neighbors:
            p2 = np.array(camerasInfo[n]['position'])[:2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c='red', lw=0.8, alpha=0.5)

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("Non-overlapping Images and Neighbor Relationships")
    ax.legend()
    ax.axis("equal")

    save_path = os.path.join(output_path, "camera_distribution.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[VIZ] Saved visualization to {save_path}")


global_camerasInfo = None

def init_worker(cameras_info_share):
    """
    初始化子进程，将 camerasInfo 设为全局变量。
    避免每次任务都 pickle 传输巨大的字典。
    """
    global global_camerasInfo
    global_camerasInfo = cameras_info_share

def process_single_image(args):
    """
    单个图片的处理逻辑（Worker 函数）
    """
    img_id, nimg_ids, depth_path, output_path = args
    
    # 从全局变量获取 info
    cam_dict = global_camerasInfo[img_id]
    
    # img_name = cam_dict['img_name'].split('/')[-1] # 如果不需要打印或调试，这行可以注释掉节省时间
    width = int(cam_dict['width'])
    height = int(cam_dict['height'])
    
    # 确定是.png还是.jpg
    # 优化：提前构建路径，减少字符串操作
    base_name = os.path.join(depth_path, cam_dict['img_name'])
    png_path = base_name + '.png.geometric.bin'
    
    if os.path.exists(png_path):
        depth = read_depth(png_path)
    else:
        # 假设如果不是png就是jpg，或者在这里加一个check
        depth = read_depth(base_name + '.jpg.geometric.bin')

    overlap_area_all = np.zeros((height, width))
    
    for nimg_id in nimg_ids:
        # 跳过当前视角
        if int(nimg_id) == int(img_id):
            continue
            
        ncam_dict = global_camerasInfo[nimg_id]
        nwidth = int(ncam_dict['width'])
        nheight = int(ncam_dict['height'])

        # 重叠区域 (假设 grid_reprojection_ 是线程安全或纯函数)
        overlap_mask = grid_reprojection_([nheight, nwidth], ncam_dict, [height, width], depth, cam_dict)
        
        # 添加到总体的重叠中
        overlap_area_all = overlap_area_all + overlap_mask

    
    # 存储overlap_area_all
    overlap_save_path = os.path.join(output_path, f"overlap_mask_{img_id}.npy")
    np.save(overlap_save_path, overlap_area_all.astype(bool))
    
    
    # 判别总体与其他视角的重叠区域是否超过vis_thred
    # 优化：利用 numpy 快速求和
    overlap_total_num = np.count_nonzero(overlap_area_all > 0) # 比 sum(sum(...)) 更快
    
    # 注意：原逻辑是 total / area >= vis_thred (95)。
    # 通常比率是 0.0-1.0，阈值是 0.95。如果是 95，请确认 vis_thred 的输入单位。
    # 这里保持原逻辑不变。
    ratio = overlap_total_num / (width * height)
    
    return img_id, ratio

def find_vis_in_neighbor(camerasInfo, overlap_images, depth_path, output_path, vis_thred=0.95):
    """
    并行版本的可视区域查找函数
    """
    image_overlap_ratios = {}
    overlap_mask_path = os.path.join(output_path, 'overlap_mask')
    os.makedirs(overlap_mask_path, exist_ok=True)
    
    # 准备参数列表
    tasks = []
    for img_id, nimg_ids in overlap_images.items():
        tasks.append((int(img_id), nimg_ids, depth_path, overlap_mask_path))

    # 设置进程数，默认使用 CPU 核心数 - 4，防止卡死系统
    num_processes = max(1, cpu_count() - 4)
    
    print(f"Starting parallel processing with {num_processes} processes...")

    # 启动进程池
    # 使用 initializer 传递 camerasInfo，避免每次任务都复制它
    with Pool(processes=num_processes, initializer=init_worker, initargs=(camerasInfo,)) as pool:
        # 使用 imap_unordered 稍微快一点，因为我们不关心结果回来的顺序
        # chunksize 可以根据任务数量微调，默认即可
        results = list(tqdm(
            pool.imap_unordered(process_single_image, tasks, chunksize=1), 
            total=len(tasks), 
            desc="Correspondence Establishment (Parallel)"
        ))

    # 将并行结果收集到字典中
    for img_id, ratio in results:
        image_overlap_ratios[img_id] = ratio

    print(f'Finished calculating ratios for {len(image_overlap_ratios)} images.')

    # --- 如果需要保存到 output_path (例如保存为 txt) ---
    if output_path:
        # 按照 img_id 排序后保存
        sorted_ids = sorted(image_overlap_ratios.keys())
        with open(os.path.join(output_path, 'overlap_ration.txt'), 'w') as f:
            for img_id in sorted_ids:
                # 格式: img_id ratio
                f.write(f"{img_id} {image_overlap_ratios[img_id]:.6f}\n")
        print(f"Ratios saved to {output_path}")

    return image_overlap_ratios


def main():
    ####################################### 参数 #######################################
    workspace = r"/home/rylynn/Pictures/LinesDetection_Workspace/datasets/Dublin/block2/"
    image_scale = 1
    overlap_percentile = 50       # 自动阈值分位数,可以简单理解为取前%为重叠航片
    area_ratio_th = 0.5           # 面积重叠比例阈值
    dist_th = 25.0                # 相机中心距离阈值（米）

    ####################################### 路径 #######################################
    sparse_model_path = os.path.join(workspace, 'sparse')
    images_path=os.path.join(workspace, 'images')
    output_path = os.path.join(workspace, 'intermediate_results_0118')
    depthmap_orginal_path = "/media/rylynn/data/Dublin/block2/dense/stereo/depth_maps/"
    depthmap_path = os.path.join(workspace, 'depth_maps')
    gt_images_path = os.path.join(output_path, 'gt', 'images')
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(depthmap_path, exist_ok=True)
    os.makedirs(gt_images_path, exist_ok=True)

    ####################################### Step 1: 加载稀疏模型 #######################################
    camerasInfo, points_in_images = load_sparse_model(sparse_model_path, image_scale)
    print(f"[INFO] Loaded {len(camerasInfo)} images.")

    json_files = [f for f in os.listdir(output_path) if f.startswith("near_image_ids_") and f.endswith(".json")]
    # 如果存在则直接读取
    if len(json_files)>0:
        print(f"[INFO] near_image_ids file exists in {output_path}, skipping selection.")
        with open(os.path.join(output_path, json_files[0]), "r") as f:
            near_image_ids = json.load(f)
        # 确保 key 是整数
        near_image_ids = {int(k):v for k,v in near_image_ids.items()}
        
    else:
        ####################################### Step 2: 动态阈值 #######################################
        print("[INFO] Computing match statistics...")
        match_counts = []
        # 计算匹配点矩阵
        matches_matrix,_ = match_pair(camerasInfo, points_in_images)
        # 统计匹配点数量（剔除0）
        match_counts = matches_matrix[matches_matrix > 0].flatten()
        # 根据匹配点分布自适应确定阈值
        match_point_num = int(np.percentile(match_counts, 100-overlap_percentile))  # 等效于从大到小取
        print(f"[INFO] Adaptive match threshold = {match_point_num}")

        ####################################### Step 3: 构建重叠图 + 选取不重叠航片 #######################################
        print("[INFO] Building overlap graph...")
        G, overlap_images = build_overlap_graph(camerasInfo, points_in_images, match_point_num)
        # 选取不重叠航片（最大独立集）
        nonoverlap_ids = choose_nonoverlapping_images(G, camerasInfo)
        print(f"[INFO] Selected {len(nonoverlap_ids)} non-overlapping images.")

        ####################################### Step 4: 查找邻近航片 #######################################
        _, overlap_images = match_pair(camerasInfo, points_in_images, match_point_num=match_point_num)
        near_image_ids = {}
        for img1_id in tqdm(nonoverlap_ids):
            #cam_dict1 = camerasInfo[img1_id]
            #lens1 = cam_dict1['img_name'].split('/')[0]
            #near_images = []
            near_images = overlap_images[img1_id]
            '''
            for img2_id in overlap_images.get(img1_id, []):
                cam_dict2 = camerasInfo[img2_id]
                lens2 = cam_dict2['img_name'].split('/')[0]
                dist = np.linalg.norm(np.array(cam_dict1['position']) - np.array(cam_dict2['position']))
                if dist > dist_th:
                    continue
                common_points = find_common_points(img1_id, img2_id, camerasInfo)
                if common_points is None or len(common_points) < 10:
                    continue
                r1, r2 = compute_overlap_ratio(cam_dict1, cam_dict2, common_points)
                # 保证两个相片的镜头相同（避免大的偏移）
                if r1 > area_ratio_th and r2 > area_ratio_th and lens1 == lens2:
                    near_images.append(int(img2_id))
            '''
            near_image_ids[img1_id] = near_images

        # 剔除少于3张邻近航片的航片
        near_image_ids = {k:v for k,v in near_image_ids.items() if len(v) > 3}

        ####################################### Step 5: 保存结果 #######################################
        json_path = os.path.join(output_path, f"near_image_ids_{match_point_num}.json")
        with open(json_path, 'w') as f:
            json.dump(near_image_ids, f, indent=2)
        print(f"[INFO] Saved near-image dictionary to {json_path}")

        ####################################### Step 6: 可视化 #######################################
        visualize_camera_distribution(camerasInfo, nonoverlap_ids, near_image_ids, output_path)

        print(f"[STATS] Avg. neighbors per image: {np.mean([len(v) for v in near_image_ids.values()]):.2f}")

    ####################################### Step 7: 拷贝需要的depthmap到路径中 #######################################
    # 统计需要深度图的航片
    images_depth = set()
    for id, near_images in near_image_ids.items():
        images_depth.add(id)
        for iid in near_images:
            images_depth.add(iid)
    # 拷贝深度图到路径
    for cam_id in tqdm(images_depth):
        cam_dict = camerasInfo[cam_id]
        depth_path = os.path.join(depthmap_orginal_path, cam_dict['img_name']+'.jpg.geometric.bin')
        new_depth_path = os.path.join(depthmap_path, cam_dict['img_name']+'.jpg.geometric.bin')
        # 拷贝文件从depth_path到new_depth_path
        if os.path.exists(depth_path):
            os.makedirs(os.path.dirname(new_depth_path), exist_ok=True)
            if not os.path.exists(new_depth_path):
                os.system(f'cp "{depth_path}" "{new_depth_path}"')
    print(f"[INFO] Copied required depth maps to {depthmap_path}")

    ####################################### Step 8: 计算邻近视角的overlap_mask #######################################
    find_vis_in_neighbor(camerasInfo, near_image_ids, depthmap_path, output_path)

    with open(os.path.join(output_path, 'overlap_ration.txt'), 'r') as f:
        lines = f.readlines()
    overlap_ratios = {}
    for line in lines:
        img_id, ratio = line.strip().split()
        overlap_ratios[int(img_id)] = float(ratio)
    for img_id in near_image_ids:
        img_name = camerasInfo[img_id]['img_name'].split('/')[1]
        # 读取邻近视角overlap mask
        overlap_mask_path = os.path.join(output_path, "overlap_mask", f"overlap_mask_{img_id}.npy")
        overlap_mask = np.load(overlap_mask_path)
        # 可视化mask
        r = math.floor(overlap_ratios[img_id]*1000)
        plt.imsave(os.path.join(output_path, 'overlap_mask_viz',f'{img_name}_ratio{r}.jpg'), overlap_mask, cmap='binary')
 

    ####################################### Step 9: 拷贝需要的images #######################################
    for img_id in near_image_ids:
        cam_dict = camerasInfo[img_id]
        img_path = os.path.join(images_path, cam_dict['img_name']+'.jpg')
        new_img_path = os.path.join(gt_images_path, cam_dict['img_name'].split('/')[1]+'.jpg')
        if os.path.exists(img_path):
            if not os.path.exists(new_img_path):
                os.system(f'cp "{img_path}" "{new_img_path}"')
    print(f"[INFO] Copied required images to {gt_images_path}")        
if __name__ == "__main__":
    main()