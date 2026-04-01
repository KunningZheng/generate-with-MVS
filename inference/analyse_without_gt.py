import os
from datasets.line3dpp_loader import parse_lines3dpp
import numpy as np

if __name__ == "__main__":
    ####################################### 参数 #######################################
    workspace = r"/home/rylynn/Pictures/LinesDetection_Workspace/datasets/Dublin_block3/sparse_txt_geo/"
    overlap_percentile = 90       # 自动阈值分位数,可以简单理解为取前%为重叠航片

    ####################################### 路径 #######################################
    output_path = os.path.join(workspace, 'intermediate_results_0104')
    line3d_len3000_path = os.path.join(workspace, 'Line3D++_H_strict')
    line3d_matched_path = os.path.join(workspace, 'Line3D++_H_matched_normal')


    # 1. 读取Line3D++的重建的重建结果，记录各相片实际参与重建的线段结果，记录各相片实际参与重建的线段
    line3d_len, line3d_to_line2d_len,_ = parse_lines3dpp(line3d_len3000_path)
    line3d_matched, line3d_to_line2d_matched,_ = parse_lines3dpp(line3d_matched_path)


    def analyze_lines_statistics(tag, lines3d, lines3d_to_2d):
        print(f"========== 统计任务: {tag} ==========")
        
        # 预处理：计算所有线段长度
        # lines3d 形状为 (N, 6), 每一行是 (x1, y1, z1, x2, y2, z2)
        # 计算 (x1-x2, y1-y2, z1-z2)
        diff_vectors = lines3d[:, :3] - lines3d[:, 3:]
        # 计算欧氏距离 (L2 norm)
        lengths = np.linalg.norm(diff_vectors, axis=1)

        # 预处理：计算每条3D线段的匹配数
        match_counts = [len(matches) for matches in lines3d_to_2d.values()]

        # 1. 线段平均长度
        avg_len = np.mean(lengths)
        print(f"1. 线段平均长度: {avg_len:.4f}")

        # 2. 线段长度分布 (使用分位数展示: Min, 25%, Median, 75%, Max)
        len_dist = np.percentile(lengths, [0, 25, 50, 75, 100])
        print(f"2. 线段长度分布 (0%, 25%, 50%, 75%, 100%): {len_dist}")

        # 3. 每条三维线段匹配的二维线段平均数量
        if match_counts:
            avg_matches = np.mean(match_counts)
            print(f"3. 每条三维线段匹配的二维线段平均数量: {avg_matches:.4f}")

            # 4. 每条三维线段匹配的二维线段数量分布
            match_dist = np.percentile(match_counts, [0, 25, 50, 75, 100])
            print(f"4. 每条三维线段匹配的二维线段数量分布 (0%, 25%, 50%, 75%, 100%): {match_dist}")
        else:
            print("3 & 4. 无匹配数据")
        print("\n")

    # 执行统计 (针对读取的两个结果)
    analyze_lines_statistics("Results: H_strict_len3000", line3d_len, line3d_to_line2d_len)
    analyze_lines_statistics("Results: H_matched_normal_len3000", line3d_matched, line3d_to_line2d_matched)
