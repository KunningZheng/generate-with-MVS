import numpy as np
import cv2
from pytlsd import lsd
from afm_op import afm
import torch
from skimage.draw import line
from tqdm import tqdm

def rasterize_lines(image_shape, lines):
    """
    将线段栅格化，生成一个与图片同尺寸的 numpy 数组，每个像素存储线段编号。
    
    参数：
    - image_shape: (H, W) 代表输出栅格的高度和宽度
    - lines: 线段列表，每条线段的格式为 (x1, y1, x2, y2)
    
    返回：
    - raster: 2D numpy 数组，与 image_shape 相同，包含线段编号，未被线段覆盖的像素为 -1
    """
    H, W = image_shape
    raster_lines = np.full((H, W), -1, dtype=int)  # 初始化栅格，未被覆盖的像素设为 -1
    
    for idx, (y1, x1, y2, x2) in enumerate(lines):
        # 计算线段的像素点
        rr, cc = line(round(y1), round(x1), round(y2), round(x2))  # skimage.draw.line 返回行列索引（y, x）
        
        # 过滤掉超出范围的点
        valid_idx = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        rr, cc = rr[valid_idx], cc[valid_idx]
        
        # 在栅格上标记线段编号
        raster_lines[rr, cc] = idx

    return raster_lines


def clip_lines_to_image(lines, height, width):
    '''
    将LSD检测到的线段裁剪到图像范围内
    - 参数
        - lines: np.ndarray, 形状为(N, 2, 2), 每一行表示一个线段[[x1, y1],[x2, y2]],x从上到下,y从左到右
        - height: 图像长度
        - width: 图像宽度
    - 返回
        - clipped_lines: np.ndarray, 形状为(N, 2, 2)
    '''
    lines = lines.reshape((-1, 4))
    clipped_lines = np.clip(lines,[0, 0, 0, 0],[height-1, width-1, height-1, width-1])
    return clipped_lines.reshape((-1, 2, 2))


def af_df_producer(lines, img):
    h, w = img.shape[:2]
    pix_loc = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='ij'),
                       axis=-1)
    raster_lines = np.zeros_like(img)

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

    # Get raster_lines   
    raster_lines = (df < 1).astype(np.uint8)
    raster_lines = np.where(raster_lines > 0, np.ones_like(img),
                            np.zeros_like(img))    
    
    return df, angle, closest, raster_lines    


def select_lines(img, nlines, raster_lines, retain_ratio):
    retained_ids = []
    
    for idx, line in enumerate(nlines):
        x1, y1 = line[0]
        x2, y2 = line[1]
        
        # 计算线段长度（像素数）
        length = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
        if length == 0:
            continue
        
        # 生成线段上的采样点
        sampled_points = []
        for i in range(length + 1):
            t = i / length
            x = int(x1 * (1 - t) + x2 * t)
            y = int(y1 * (1 - t) + y2 * t)
            
            # 检查是否在图像范围内
            if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
                sampled_points.append((x, y))
        
        if not sampled_points:
            continue
        
        # 统计在raster_lines区域内的点数
        points_in_region = 0
        for x, y in sampled_points:
            if raster_lines[x, y] > 0:  # 假设raster_lines是二值图，0表示背景，>0表示线段区域
                points_in_region += 1
        
        # 计算比例
        ratio = points_in_region / len(sampled_points)
        
        # 如果满足阈值，则保留该线段
        if ratio >= retain_ratio:
            retained_ids.append(idx)
    retained_nlines = nlines[retained_ids]
    return retained_nlines



def establish_line_correspondences(lines, nlines, 
                                   t_angle=10.0, 
                                   t_dist=10.0, 
                                   t_overlap=0.3):
    """
    Establishes matching relationship between current view lines and projected neighbor lines
    based on the Geometric Similarity Measurement described in the paper.
    
    Paper Source: "3-D Line Segment Reconstruction With Depth Maps for Photogrammetric Mesh Refinement"
                  Section III-A-1, Eq (3)-(4), Table II.

    Args:
        lines: numpy array of shape (N, 2, 2). Current view lines (candidates, l_s).
               Format: [[x1, y1], [x2, y2]] per line.
        nlines: numpy array of shape (M, 2, 2). Neighbor view projected lines (virtual lines, v_r).
        t_angle: Angle threshold in degrees (default 10 per Table II).
        t_dist: Perpendicular distance threshold in pixels (default 10 per Table II).
        t_overlap: Overlap threshold (default 0.3 per Table II).

    Returns:
        matches: List of tuples (nline_idx, line_idx, score).
                 Indicates nlines[nline_idx] matches lines[line_idx] with similarity score.
    """
    
    # 1. Pre-calculate vectors and lengths
    # Vector for lines (N, 2)
    vec_lines = lines[:, 1, :] - lines[:, 0, :]
    len_lines = np.linalg.norm(vec_lines, axis=1)
    
    # Vector for nlines (M, 2)
    vec_nlines = nlines[:, 1, :] - nlines[:, 0, :]
    len_nlines = np.linalg.norm(vec_nlines, axis=1)
    
    # Avoid division by zero
    len_lines[len_lines == 0] = 1e-6
    len_nlines[len_nlines == 0] = 1e-6
    
    # Unit vectors
    unit_lines = vec_lines / len_lines[:, None]
    unit_nlines = vec_nlines / len_nlines[:, None]
    
    matches = []
    
    # Iterate over each projected line (reference/virtual line)
    # The paper describes finding the best match for each virtual line v_r from candidates l_s
    for i in range(len(nlines)):
        v_r = nlines[i]
        u_v = unit_nlines[i]
        l_v = len_nlines[i]
        
        # --- A. Angle Similarity (d_alpha) ---
        # Compute dot product between current v_r and all candidate lines
        # shape: (N,)
        dots = np.abs(np.sum(unit_lines * u_v, axis=1))
        dots = np.clip(dots, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(dots))
        
        # d_alpha = 1 - alpha / T_alpha (Eq 4)
        d_alpha = 1 - angles_deg / t_angle
        
        # Filter 1: Angle constraint
        valid_angle_mask = d_alpha >= 0
        
        if not np.any(valid_angle_mask):
            continue
            
        # Optimization: Only process candidates that pass angle test
        candidate_indices = np.where(valid_angle_mask)[0]
        
        scores = []
        
        for j in candidate_indices:
            l_s = lines[j]
            l_cand = len_lines[j]
            
            # --- B. Distance Similarity (d1, d2) ---
            # Perpendicular distance from endpoints of l_s (candidate) to line of v_r (virtual)
            # Line v_r defined by point v_r[0] and direction u_v
            # Dist = |(P - P0) cross U| (2D cross product is determinant)
            
            # Vector from v_r[0] to l_s endpoints
            vec_p1 = l_s[0] - v_r[0]
            vec_p2 = l_s[1] - v_r[0]
            
            # Cross product in 2D: x1*y2 - x2*y1
            cross1 = vec_p1[0] * u_v[1] - vec_p1[1] * u_v[0]
            cross2 = vec_p2[0] * u_v[1] - vec_p2[1] * u_v[0]
            
            dist1 = np.abs(cross1)
            dist2 = np.abs(cross2)
            
            # d1 = 1 - v1 / T_d (Eq 4)
            d1 = 1 - dist1 / t_dist
            d2 = 1 - dist2 / t_dist
            
            if d1 < 0 or d2 < 0:
                continue
                
            # --- C. Overlap Similarity (d_o) ---
            # Project l_s onto the line of v_r to measure overlap
            # We define coordinate system along v_r starting at v_r[0]
            # Projection of point P is (P - P0) dot U
            proj_v_start = 0.0
            proj_v_end = l_v
            
            proj_l_start = np.dot(l_s[0] - v_r[0], u_v)
            proj_l_end = np.dot(l_s[1] - v_r[0], u_v)
            
            # Ensure l_start < l_end for interval logic
            if proj_l_start > proj_l_end:
                proj_l_start, proj_l_end = proj_l_end, proj_l_start
                
            # Intersection of [0, l_v] and [proj_l_start, proj_l_end]
            inter_start = max(0.0, proj_l_start)
            inter_end = min(l_v, proj_l_end)
            
            length_o = max(0.0, inter_end - inter_start)
            
            # d_o = Length_o / min(Length_v, Length_l) / T_o - 1 (Eq 4)
            min_len = min(l_v, l_cand)
            if min_len < 1e-6:
                d_o = -1
            else:
                d_o = (length_o / min_len) / t_overlap - 1
                
            if d_o < 0:
                continue
                
            # --- Final Similarity ---
            # Sim = d_alpha + d1 + d2 (Eq 3, if overlap condition met)
            sim = d_alpha[j] + d1 + d2
            scores.append((j, sim))
        
        # Select best match for this virtual line
        if scores:
            # Sort by score descending
            scores.sort(key=lambda x: x[1], reverse=True)
            best_idx, best_score = scores[0]
            matches.append((i, best_idx, best_score))

    return matches


def establish_line_correspondences_reverse(lines, nlines, 
                                   t_angle=10.0, 
                                   t_dist=10.0, 
                                   t_overlap=0.3):
    """
    Establishes matching relationship between current view lines and projected neighbor lines
    based on the Geometric Similarity Measurement described in the paper.
    
    Paper Source: "3-D Line Segment Reconstruction With Depth Maps for Photogrammetric Mesh Refinement"
                  Section III-A-1, Eq (3)-(4), Table II.

    Args:
        lines: numpy array of shape (N, 2, 2). Current view lines (candidates, l_s).
               Format: [[x1, y1], [x2, y2]] per line.
        nlines: numpy array of shape (M, 2, 2). Neighbor view projected lines (virtual lines, v_r).
        t_angle: Angle threshold in degrees (default 10 per Table II).
        t_dist: Perpendicular distance threshold in pixels (default 10 per Table II).
        t_overlap: Overlap threshold (default 0.3 per Table II).

    Returns:
        matches: List of tuples (line_idx, nline_idx, score).
                        Indicates lines[line_idx] matches nlines[nline_idx] with similarity score.
    """
    
    # 1. Pre-calculate vectors and lengths
    # Vector for lines (N, 2)
    vec_lines = lines[:, 1, :] - lines[:, 0, :]
    len_lines = np.linalg.norm(vec_lines, axis=1)
    
    # Vector for nlines (M, 2)
    vec_nlines = nlines[:, 1, :] - nlines[:, 0, :]
    len_nlines = np.linalg.norm(vec_nlines, axis=1)
    
    # Avoid division by zero
    len_lines[len_lines == 0] = 1e-6
    len_nlines[len_nlines == 0] = 1e-6
    
    # Unit vectors
    unit_lines = vec_lines / len_lines[:, None]
    unit_nlines = vec_nlines / len_nlines[:, None]
    
    matches = []
    
    # Iterate over each CURRENT view line (Reference)
    # Changed: range(len(nlines)) -> range(len(lines))
    for i in range(len(lines)):
        # Ref: The current view line
        l_ref_seg = lines[i]       # coords
        u_ref = unit_lines[i]      # unit vector
        len_ref = len_lines[i]     # length
        
        # --- A. Angle Similarity (d_alpha) ---
        # Compute dot product between current Ref (lines[i]) and ALL candidates (nlines)
        # shape: (M,)
        dots = np.abs(np.sum(unit_nlines * u_ref, axis=1))
        dots = np.clip(dots, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(dots))
        
        # d_alpha = 1 - alpha / T_alpha
        d_alpha = 1 - angles_deg / t_angle
        
        # Filter 1: Angle constraint
        valid_angle_mask = d_alpha >= 0
        
        if not np.any(valid_angle_mask):
            continue
            
        # Optimization: Only process candidates (nlines) that pass angle test
        candidate_indices = np.where(valid_angle_mask)[0]
        
        scores = []
        
        for j in candidate_indices:
            # Cand: The neighbor projected line
            l_cand_seg = nlines[j]
            len_cand = len_nlines[j]
            
            # --- B. Distance Similarity (d1, d2) ---
            # Perpendicular distance from endpoints of CANDIDATE (nlines[j]) 
            # to the infinite line defined by REFERENCE (lines[i])
            
            # Vector from Ref start (lines[i][0]) to Cand endpoints
            vec_p1 = l_cand_seg[0] - l_ref_seg[0]
            vec_p2 = l_cand_seg[1] - l_ref_seg[0]
            
            # Cross product with Ref direction (u_ref)
            cross1 = vec_p1[0] * u_ref[1] - vec_p1[1] * u_ref[0]
            cross2 = vec_p2[0] * u_ref[1] - vec_p2[1] * u_ref[0]
            
            dist1 = np.abs(cross1)
            dist2 = np.abs(cross2)
            
            d1 = 1 - dist1 / t_dist
            d2 = 1 - dist2 / t_dist
            
            if d1 < 0 or d2 < 0:
                continue
                
            # --- C. Overlap Similarity (d_o) ---
            # Project CANDIDATE (nlines[j]) onto REFERENCE (lines[i])
            # Coordinate system based on lines[i], range [0, len_ref]
            
            proj_cand_start = np.dot(l_cand_seg[0] - l_ref_seg[0], u_ref)
            proj_cand_end = np.dot(l_cand_seg[1] - l_ref_seg[0], u_ref)
            
            if proj_cand_start > proj_cand_end:
                proj_cand_start, proj_cand_end = proj_cand_end, proj_cand_start
                
            # Intersection of Reference [0, len_ref] and Projected Candidate
            inter_start = max(0.0, proj_cand_start)
            inter_end = min(len_ref, proj_cand_end)
            
            length_o = max(0.0, inter_end - inter_start)
            
            # d_o calculation
            min_len = min(len_ref, len_cand)
            if min_len < 1e-6:
                d_o = -1
            else:
                d_o = (length_o / min_len) / t_overlap - 1
                
            if d_o < 0:
                continue
                
            # --- Final Similarity ---
            # Sim = d_alpha + d1 + d2
            # Note: d_alpha is array, need index j
            sim = d_alpha[j] + d1 + d2
            scores.append((j, sim))
        
        # Select best nline match for this current line
        if scores:
            scores.sort(key=lambda x: x[1], reverse=True)
            best_idx, best_score = scores[0]
            # Returns: (line_idx, nline_idx, score)
            matches.append((i, best_idx, best_score))

    return matches


def lsd_opencv(img):
    '''
    模仿line3D++中lsd_opencv.cpp，获取线段
    return:
        filtered_lines: np.array(N, 4)
    '''
    # 1. 调整参数：将 density_th 设置为 0.7 (与 lsd_opencv 一致)
    # 注意：scale 和 sigma_scale 默认值在 PYAPI.cpp 中已经是 0.8 和 0.6，与 lsd_opencv 一致
    lines = lsd(img, density_th=0.7)

    # 2. 模拟 line3D 的长度过滤 (L3D_DEF_MIN_LINE_LENGTH_FACTOR 为 0.005)
    # line3D.cc 中计算长度: sqrt(dx*dx + dy*dy)
    h, w = img.shape[:2]
    diag = np.sqrt(h**2 + w**2)
    min_len = diag * 0.005

    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2, _ = line
        length = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        if length > min_len:
            filtered_lines.append(line)

    filtered_lines = np.array(filtered_lines)

    return filtered_lines