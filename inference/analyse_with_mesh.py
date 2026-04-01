import numpy as np
import os
from scipy.spatial import cKDTree
from plyfile import PlyData
from tqdm import tqdm

from datasets.line3dpp_loader import parse_lines3dpp

class LineEvaluator:
    def __init__(self, gt_ply_path):
        """
        初始化评估器，加载真值点云并构建 KD-Tree
        """
        print(f"Loading GT point cloud from {gt_ply_path}...")
        self.gt_points = self.read_ply(gt_ply_path)
        print(f"Building KD-Tree for {self.gt_points.shape[0]} points...")
        self.tree = cKDTree(self.gt_points)
        print("KD-Tree built.")

    def read_ply(self, fname):
        """读取 .ply 文件转换为 numpy array"""
        plydata = PlyData.read(fname)
        x = np.asarray(plydata.elements[0].data["x"])
        y = np.asarray(plydata.elements[0].data["y"])
        z = np.asarray(plydata.elements[0].data["z"])
        return np.stack([x, y, z], axis=1)

    def sample_points_on_line(self, line_start, line_end, sample_step=0.01):
        """
        在线段上均匀采样点
        line_start, line_end: (3,) numpy array
        sample_step: 采样步长（单位：米），建议 0.01 (1cm) 或 0.001 (1mm)
        """
        vec = line_end - line_start
        length = np.linalg.norm(vec)
        
        if length < 1e-6:
            return np.array([line_start]), length
        
        # 计算采样点数量
        n_samples = int(np.ceil(length / sample_step)) + 1
        
        # 生成 t 值 [0, 1]
        t = np.linspace(0, 1, n_samples)
        
        # 插值生成点: P = A + t * (B - A)
        sampled_points = line_start[None, :] + t[:, None] * vec[None, :]
        return sampled_points, length

    def compute_metrics(self, lines, thresholds_mm=[5, 10, 20], sample_step=0.01):
        """
        计算主要指标
        lines: 列表，每个元素包含 {'geometry': (2,3) np array, 'supports': (n_img, n_lines)}
        thresholds_mm: 阈值列表，单位毫米
        sample_step: 采样步长，单位米
        """
        # 转换阈值为米
        thresholds_m = [t / 1000.0 for t in thresholds_mm]
        
        total_length = 0
        list_ratios = {th: [] for th in thresholds_m}
        list_lengths = []
        
        # --- 1. 遍历每一条线进行距离计算 ---
        print("Evaluating lines...")
        for line_data in tqdm(lines):
            p_start = line_data['geometry'][0]
            p_end = line_data['geometry'][1]
            
            # 步骤 A: 采样
            samples, length = self.sample_points_on_line(p_start, p_end, sample_step)
            list_lengths.append(length)
            
            # 步骤 B: 查询 KD-Tree 计算最近距离
            # k=1 返回最近的一个点 (distance, index)
            dists, _ = self.tree.query(samples, k=1, workers=-1) # workers=-1 使用所有CPU核心
            
            # 步骤 C: 针对不同阈值计算 Inlier Ratio
            for th in thresholds_m:
                # 统计距离小于阈值的采样点比例
                inlier_ratio = np.sum(dists <= th) / len(dists)
                list_ratios[th].append(inlier_ratio)

        list_lengths = np.array(list_lengths)
        
        # --- 2. 汇总指标 ---
        results = {}
        
        for idx, th in enumerate(thresholds_m):
            th_mm = thresholds_mm[idx]
            ratios = np.array(list_ratios[th])
            
            # 指标 1: Length Recall (R_tau)
            # 公式: sum(length * ratio) / sum(total_gt_length) 
            # 注意: Limap 代码中的 recall 实际上是绝对长度 (meters)，不是比例
            # 这里按照你提供的代码逻辑: (lengths * ratios).sum()
            length_recall = (list_lengths * ratios).sum()
            
            # 指标 2: Inlier Percentage (P_tau)
            # 公式: percentage of tracks that are within tau (ratios > 0)
            # 你提供的代码逻辑: (ratios > 0).sum() / count
            precision = 100 * (ratios > 0).astype(float).sum() / len(lines)
            
            results[th_mm] = {
                "R_tau_meters": length_recall,
                "P_tau_percent": precision
            }

        # --- 3. 计算 Average Supports ---
        # 假设 line_data 包含 'n_img_supports' 和 'n_line_supports'
        avg_img_supports = np.mean([l.get('n_img_supports', 0) for l in lines])
        avg_line_supports = np.mean([l.get('n_line_supports', 0) for l in lines])
        
        return results, (avg_img_supports, avg_line_supports)

# ================= 使用示例 =================

def load_your_lines(folder_path):
    """
    [需要你自己修改] 
    请根据你的实际数据格式修改此函数。
    假设你的线段存储在 numpy 或 txt 中。
    这里模拟了一些假数据。
    """
    # 1. 调用解析函数
    raw_lines3d, lines3d_to_lines2d, _ = parse_lines3dpp(folder_path)
    
    formatted_lines = []
    num_lines = raw_lines3d.shape[0]
    
    print(f"Adapting {num_lines} lines for evaluation...")
    
    for i in range(num_lines):
        # 提取几何信息 (x1, y1, z1, x2, y2, z2) -> (2, 3) 矩阵
        seg = raw_lines3d[i]
        geometry = np.array([
            [seg[0], seg[1], seg[2]],
            [seg[3], seg[4], seg[5]]
        ])
        
        # 提取 Supports 信息
        supports_list = lines3d_to_lines2d.get(i, [])
        
        # 指标3要求：
        # Image supports: 多少个不同的 cam_id
        unique_cams = set([item[0] for item in supports_list])
        n_img_supports = len(unique_cams)
        
        # 2D line supports: 总共有多少个 2D 线段支持
        n_line_supports = len(supports_list)
        
        formatted_lines.append({
            'geometry': geometry,
            'n_img_supports': n_img_supports,
            'n_line_supports': n_line_supports
        })
        
    return formatted_lines

if __name__ == "__main__":
    # 配置路径
    GT_POINT_CLOUD_PATH = "/media/rylynn/data/MatrixCity/matrixcity_point_cloud_ds/point_cloud/aerial/Block_B.ply" # 修改为你的 .ply 路径
    PRED_LINES_PATH = "/home/rylynn/Pictures/datasets_3Dline/MatrixCity/block_B/sparse_txt/Line3D++/"        # 修改为你的线段文件路径
    
    # 1. 准备数据
    lines = load_your_lines(PRED_LINES_PATH) # 取消注释并实现加载函数

    # 2. 开始评估
    evaluator = LineEvaluator(GT_POINT_CLOUD_PATH)
    
    # 定义阈值 (毫米)
    thresholds = [1, 5, 10] # 5mm, 10mm, 20mm
    
    # 计算
    metrics, supports = evaluator.compute_metrics(lines, thresholds_mm=thresholds)
    
    # 3. 打印报告
    print("\n========= Evaluation Report =========")
    print(f"Number of Line Tracks: {len(lines)}")
    print(f"Average Image Supports: {supports[0]:.2f}")
    print(f"Average 2D Line Supports: {supports[1]:.2f}")
    print("-" * 30)
    for th in thresholds:
        print(f"Threshold {th} mm:")
        print(f"  Length Recall (R): {metrics[th]['R_tau_meters']:.4f} meters")
        print(f"  Inlier Percentage (P): {metrics[th]['P_tau_percent']:.2f} %")
    print("=====================================")