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


def main():
    ####################################### 参数 #######################################
    workspace = r"/home/rylynn/Pictures/LinesDetection_Workspace/datasets/Dublin/block2/"
    image_scale = 1
    overlap_percentile = 50       # 自动阈值分位数,可以简单理解为取前%为重叠航片
    area_ratio_th = 0.5           # 面积重叠比例阈值
    dist_th = 25.0                # 相机中心距离阈值（米）
    n_jobs = 8                    # 并行线程数

    ####################################### 路径 #######################################
    sparse_model_path = os.path.join(workspace, 'sparse')
    output_path = os.path.join(workspace, 'intermediate_results')
    depthmap_orginal_path = "/media/rylynn/data/Dublin/block2/dense/stereo/depth_maps/"
    depthmap_path = os.path.join(workspace, 'depth_maps')
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(depthmap_path, exist_ok=True)

    ####################################### Step 1: 加载稀疏模型 #######################################
    camerasInfo, points_in_images = load_sparse_model(sparse_model_path, image_scale)
    print(f"[INFO] Loaded {len(camerasInfo)} images.")

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
    near_image_ids = {}
    for img1_id in tqdm(nonoverlap_ids):
        cam_dict1 = camerasInfo[img1_id]
        lens1 = cam_dict1['img_name'].split('/')[0]
        near_images = []
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

if __name__ == "__main__":
    main()