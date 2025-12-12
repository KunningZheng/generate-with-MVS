import numpy as np
import cv2
import os
import itertools
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

from pytlsd import lsd
from deeplsd.geometry.line_utils import clip_line_to_boundaries
from deeplsd.geometry.viz_2d import get_flow_vis, plot_images, plot_lines, save_plot
from deeplsd.datasets.utils.homographies import sample_homography, warp_lines, warp_points

from datasets.dataset_reader import read_cam_dict
from utils.visualize import viz_lines3D, viz_points2
from utils.visualize import viz_lines2D2 as viz
from utils.line_tools import clip_lines_to_image, rasterize_lines


def bilinear_interpolate(depthmap, x, y, scale=[1,1]):
    '''
    双线形插值，另外约束参与计算的整数坐标对应的深度应都不为0，否则输出0
    - 输入
        - depthmap: 深度图
        - x: 坐标x，对应深度图矩阵的行号
        - y: 坐标y，对应深度图矩阵的列号
        - scale: 航片缩小的倍数
    - 输出
        - 坐标对应的深度(一维矩阵)
    '''

    
    x = np.asarray(x) * scale[0]
    y = np.asarray(y) * scale[1]
    
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # Handle borders
    # 方法一（弃用）：会造成整数的边缘像素位置计算失败，例如（0，5985）
    #x0 = np.clip(x0, 0, depthmap.shape[0] - 1)  # x为行号，shape[0]也就是总行数
    #x1 = np.clip(x1, 0, depthmap.shape[0] - 1)
    #y0 = np.clip(y0, 0, depthmap.shape[1] - 1)  # y为列号，shape[1]也就是总列数
    #y1 = np.clip(y1, 0, depthmap.shape[1] - 1)
    # 方法二：使用复制边缘的方式扩展深度图
    depthmap = cv2.copyMakeBorder(depthmap,
                                  top=0,bottom=2, # 在底部增加一行
                                  left=0,right=2, # 在右侧增加一列
                                  borderType=cv2.BORDER_REPLICATE)

    # Get values from depthmap
    # depthmap本身是int16，相乘会出现“数据溢出”的问题
    Ia = depthmap[x0,y0].astype(np.float32)
    Ib = depthmap[x0,y1].astype(np.float32)
    Ic = depthmap[x1,y0].astype(np.float32)
    Id = depthmap[x1,y1].astype(np.float32)

    # Determine whether the value is 0
    d0 = Ia * Ib * Ic * Id  
    d0 = np.where(d0 == 0, 0, 1)  # 如果有一项0，则输出的结果为0

    # Calculate weights
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # Return weighted sum
    return d0 * (wa * Ia + wb * Ib + wc * Ic + wd * Id)


def depth_filter_lines(depthmap, lines, scale):
    '''
    获取线段的端点处的深度值，并过滤端点处深度为0的线段
    - 输入
        - depthmap: 深度图
        - lines: 线段坐标
        - scale: 缩小倍数
    - 输出
        - lines_depth(N, 2, 1): 线段端点处的深度
        - valid_lines(N, 2, 2): 符合条件的线段坐标
        - valid_mask: 一维bool矩阵
    '''
    # Extract endpoint
    x1, y1 = lines[:, 0, 0], lines[:, 0, 1]
    x2, y2 = lines[:, 1, 0], lines[:, 1, 1]

    # Get interpolated depths
    depth1 = bilinear_interpolate(depthmap, x1, y1, scale)
    depth2 = bilinear_interpolate(depthmap, x2, y2, scale)
    lines_depth = np.hstack((depth1.reshape(-1, 1), depth2.reshape(-1, 1)))

    # Check for zero depth and filter
    valid_mask = (depth1 != 0) & (depth2 != 0)
    valid_lines = lines[valid_mask, :, :]
    lines_depth = lines_depth[valid_mask, :].reshape(-1, 2, 1)
    return lines_depth, valid_lines, valid_mask


def cam2world(u, v, depth, cam_dict):
    '''
    从相机坐标投影到世界坐标
        - Args
            - u:image中的行号,可以为任意形状,计算时都会转成shape[1, N]
            - v:image中的列号,可以为任意形状,计算时都会转成shape[1, N]
            - depth:image的行列坐标对应的深度,可以为任意形状,计算时都会转成shape[1, N]
            - cam_dict:存有image的内外参的字典
        - Return
            - homo_pts3D:三维点的齐次坐标shape[4, N]
    '''
    # 1.获取相机内外参
    pos, rot, fx, fy, width, height = read_cam_dict(cam_dict)
    cx, cy = width // 2 , height // 2
    # 3.像平面坐标系(u,v)也就是(行号,列号) --> 像空间坐标系(x,y,z)
    cam_z = depth.reshape(1, -1)
    cam_x = cam_z * (v.reshape(1, -1) - cx) / fy # v--(cx,fy)-->x
    cam_y = cam_z * (u.reshape(1, -1) - cy) / fx  # u--(cy,fx)-->y
    cam_pts = np.concatenate((cam_x, cam_y, cam_z), axis=0)
    # 4.像空间坐标系 --> 世界坐标系
    # invert
    R = rot
    t = pos
    # 4x4 transformation
    c1_to_w = np.vstack((np.column_stack((R, t)), (0, 0, 0, 1)))
    # 将坐标转为齐次坐标
    homo_cam_pts = np.concatenate((cam_pts, np.ones((1, cam_pts.shape[1]))), axis=0)
    # 矩阵乘法进行坐标变换:
    homo_pts3D = np.dot(c1_to_w, homo_cam_pts)
    return homo_pts3D


def world2cam(homo_pts3D, cam_dict):
    '''
    从世界坐标投影到相机坐标
        - Args
            - homo_pts3D:三维点的齐次坐标
            - cam_dict:存有image的内外参的字典
        - Return
            - u:image中的行号,shape[N, 1]
            - v:image中的列号,shape[N, 1]
            - cam_pts2[2, None]:相机三维坐标的z值即对应深度

    '''
    # 1.获取相机内外参
    pos, rot, fx, fy, width, height = read_cam_dict(cam_dict)
    cx, cy = width // 2 , height // 2
    # 1.世界坐标系 --> 像空间坐标系
    # 加载旋转和平移参数
    R = rot.T
    t = - (rot.T @ pos)
    # 4x4 transformation
    w_to_c2 = np.vstack((np.column_stack((R, t)), (0, 0, 0, 1)))
    # 矩阵乘法进行坐标变换
    homo_cam_pts2  = np.dot(w_to_c2, homo_pts3D)
    # 将变换后的齐次坐标转换回普通坐标
    cam_pts2 = homo_cam_pts2[:3, ]
    # 2.像空间坐标系 --> 像平面坐标系   
    u = (fx * cam_pts2[1, None] / cam_pts2[2, None] + cy).reshape(-1, 1)
    v = (fy * cam_pts2[0, None] / cam_pts2[2, None] + cx).reshape(-1, 1)
    return u, v, cam_pts2[2, None]


def view_homo_transform_lsd():
    return 0


def viz_lines2D2(img, lines, output_path, name):
    '''
    
    '''
    plot_images([img], cmaps='viridis_r', dpi=100, size=40)
    plot_lines([lines[:,:, [1, 0]]], indices=range(1))
    save_plot(os.path.join(output_path, name + '.jpg'))


def point_to_line_segment_distance(pt1, pt2, pt3_array):
    """
    计算一组点到线段的垂直距离。
    
    :param pt1: 线段起点，形状为 (2,)
    :param pt2: 线段终点，形状为 (2,)
    :param pt3_array: 一组点，形状为 (M, 2)，其中 M 是点的数量
    :return: 一组点到线段的距离，形状为 (M,)
    """
    # 将点转换为 NumPy 数组
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    pt3_array = np.array(pt3_array)

    # 计算向量
    vector_pt1_pt2 = pt2 - pt1
    vector_pt1_pt3 = pt3_array - pt1

    # 计算叉积的模（即平行四边形的面积）
    cross_product = np.cross(vector_pt1_pt2, vector_pt1_pt3)

    # 计算线段的长度
    line_length = np.linalg.norm(vector_pt1_pt2)

    # 点到线段的垂直距离 = 叉积的模 / 线段的长度
    distances = np.abs(cross_product) / line_length

    return distances


def select_from_candidate_lines(image_pts, valid_pts_mask, dist_thredshold=1):
    '''
    - 输入
        - image_pts: (N, pts_in_line_num, 2)
        - valid_pts_mask: (N, pts_in_line_num), 深度有效的点
        - dist_thredshold: 距离小于这个阈值的为support point, 默认为1
    '''
    image_lines = np.zeros((image_pts.shape[0], 2, 2))
    for lineID, line_pts in enumerate(image_pts):
        # 生成所有点之间的不重复组合
        pts_num = list(range(image_pts.shape[1]))
        # 过滤出深度有效的点编号
        valid_pts_num = [num for num, is_valid in zip(pts_num, valid_pts_mask[lineID]) if is_valid]
        groups_of_pts = list(itertools.combinations(valid_pts_num, 2))
        # 初始化候选线段的存储矩阵
        candidate_lines = np.zeros((len(groups_of_pts), 8))
        candidateID = 0
        for (i,j) in groups_of_pts:
            # 记录pt1
            pt1 = line_pts[i]
            candidate_lines[candidateID, 0] = i
            candidate_lines[candidateID, 1] = pt1[0]
            candidate_lines[candidateID, 2] = pt1[1]
            # 记录pt2
            pt2 = line_pts[j]
            candidate_lines[candidateID, 3] = j
            candidate_lines[candidateID, 4] = pt2[0]
            candidate_lines[candidateID, 5] = pt2[1]
            # 记录候选线段的长度
            line_length = np.linalg.norm(pt2 - pt1)
            candidate_lines[candidateID, 6] = line_length 
            # 计算各点到当前候选线段的垂直距离
            orth_dist = point_to_line_segment_distance(pt1, pt2, line_pts)
            # 距离阈值筛选support_point的数量
            support_point = np.count_nonzero((orth_dist < dist_thredshold))
            candidate_lines[candidateID, 7] = support_point # 记录候选线段的support_point数量
            candidateID += 1
        # 从所有的candidate_lines中找到support_point数量最多的线段
        max_value = np.max(candidate_lines[:, 7])
        # 找到所有最大值的索引
        max_indices = np.where(candidate_lines[:, 7] == max_value)[0]
        if max_indices.size == 1:
            resultID = int(max_indices)
        else:
            # 在support_point数量最多的线段中，找到长度最长的线段
            max_length = 0
            for id in max_indices:
                if candidate_lines[id, 6] > max_length:
                    max_length = candidate_lines[id, 6]
                    resultID = id
        # 记录最终结果线段的端点
        image_lines[lineID, 0, :] = [candidate_lines[resultID, 1], candidate_lines[resultID, 2]]
        image_lines[lineID, 1, :] = [candidate_lines[resultID, 4], candidate_lines[resultID, 5]]
    
    return image_lines


def get_depth_from_depthmap(depthmap, points, scale=1.0, search_window=5):
    """
    从深度图中获取深度值。
    
    :param depthmap: 深度图，形状为 (H, W)，值为深度
    :param points: 坐标点矩阵，形状为 (N, 7, 2)，表示 (x, y) 坐标
    :param scale: 坐标缩放比例，默认为 1.0
    :param search_window: 窗口大小, 默认为5
    :return: 深度值矩阵，形状为 (N, 7)
    """
    N, P, _ = points.shape  # N: 线段数量，P: 每条线段的点数
    H, W = depthmap.shape  # 深度图的高度和宽度

    # 深度图根据窗口大小扩展
    step = (search_window - 1)  // 2
    depthmap = cv2.copyMakeBorder(depthmap,
                                  top=step,bottom=step, 
                                  left=step,right=step, 
                                  borderType=cv2.BORDER_REPLICATE)
 

    # 初始化深度值矩阵
    depths = np.zeros((N, P))

    # 遍历每个点
    for i in range(N):
        for j in range(P):
            # 转换为深度图中的像素坐标
            x, y = points[i, j] * scale
            # 加上step是因为深度图在左侧和上侧也扩展了
            x, y = int(round(x)) + step, int(round(y)) + step

            # 检查坐标是否在深度图范围内
            if 0 <= x < H+step and 0 <= y < W+step:
                # 最近邻取深度
                depth = depthmap[x, y]

                # 如果深度为0，检查窗口
                if depth == 0:
                    # 定义窗口范围
                    window_x = np.arange(max(0, x - step), min(H-1+step*2, x + step))
                    window_y = np.arange(max(0, y - step), min(W-1+step*2, y + step))

                    # 提取窗口内的深度值
                    window_depths = depthmap[window_x, window_y]

                    # 检查窗口内是否有非0深度值
                    non_zero_depths = window_depths[window_depths != 0]

                    if non_zero_depths.size > 0:
                        # 计算窗口内每个点到中心点的距离
                        window_coords = np.stack(np.meshgrid(window_x, window_y), axis=-1).reshape(-1, 2)
                        center_coord = np.array([x, y])
                        distances = np.linalg.norm(window_coords - center_coord, axis=1)

                        # 提取非0深度值及其对应的距离
                        non_zero_indices = np.nonzero(window_depths.reshape(-1))[0]
                        non_zero_distances = distances[non_zero_indices]
                        non_zero_depths = window_depths.reshape(-1)[non_zero_indices]

                        # 取距离最近的非0深度值
                        closest_index = np.argmin(non_zero_distances)
                        depth = non_zero_depths[closest_index]

                depths[i, j] = depth

    return depths


def interpolate_line_segments(line_segments, num_points=5):
    """
    在线段上等距离插入点。
    
    :param line_segments: 线段端点矩阵，形状为 (N, 2, 2)
    :param num_points: 在每条线段上插入的点数，默认为 5
    :return: 包含起点、插入点和终点的矩阵，形状为 (N, num_points + 2, 2)
    """
    N = line_segments.shape[0]  # 线段数量
    total_points = num_points + 2  # 包括起点和终点
    result = np.zeros((N, total_points, 2))  # 初始化结果矩阵

    for i in range(N):
        # 提取第 i 条线段的起点和终点
        start_point = line_segments[i, 0]
        end_point = line_segments[i, 1]

        # 计算线段的差值向量
        delta = end_point - start_point

        # 插值计算每个点的坐标
        for j in range(total_points):
            # 插值公式：点 = 起点 + (j / (total_points - 1)) * 差值向量
            result[i, j] = start_point + (j / (total_points - 1)) * delta

    return result


def views_transform_lsd(img, depth, cam_dict, nimg, ndepth, ncam_dict):
    '''
    邻近视角(包括当前视角)分别进行LSD，然后投影到当前视角，用深度突变边界剔除
    '''
    ################################ LSD ################################
    h, w = nimg.shape[:2]
    nimage_lines = lsd(nimg)[:, [1, 0, 3, 2]].reshape(-1, 2, 2)
    # Clip lines to image（LSD算法的线段延长特性导致线段端点超出范围）
    nimage_lines, valid = clip_line_to_boundaries(nimage_lines, nimg.shape, min_len=0)
    nimage_lines = nimage_lines[valid]

    ############################## 投影到当前视角 ##############################
    # 如果为当前视角，那就无需投影，直接返回lines
    if cam_dict['img_name'] == ncam_dict['img_name']:
        return nimage_lines
    # 其他情况则需要投影
    else:
        ## 1.在线段中插入5个点
        nimage_pts = interpolate_line_segments(nimage_lines, num_points=5)
        # test:可视化
        #from visualize.visualize import visualize_point_line_image
        #visualize_point_line_image(nimg, nimage_lines, nimage_pts)

        ## 2.赋值深度
        ndepth_scale = ndepth.shape[0] / nimg.shape[0]
        nimage_pts_depth = get_depth_from_depthmap(ndepth, nimage_pts, scale=ndepth_scale, search_window=5)
        # 剔除深度为有效值(不为0)的点数少于2的线段
        nimage_pts_depth_ = nimage_pts_depth > 0
        valid_line_mask = np.sum(nimage_pts_depth_, axis=1) > 1
        nimage_pts_depth = nimage_pts_depth[valid_line_mask]
        nimage_pts = nimage_pts[valid_line_mask]
        # 仍有深度为0的点，为了保持每条线段有7个点的条件
        valid_pts_mask = nimage_pts_depth > 0

        ## 3.投影
        # 投影到三维中
        nimage_pts_x = nimage_pts[:, :, 0]  # 行号
        nimage_pts_y = nimage_pts[:, :, 1]  # 列号  
        homo_pts3D = cam2world(nimage_pts_x, nimage_pts_y, nimage_pts_depth, ncam_dict)
        # test:可视化
        #viz_points2(homo_pts3D[:3, :].T)
        # 投影到当前视角中
        image_pts_x, image_pts_y, _= world2cam(homo_pts3D, cam_dict)
        image_pts = np.hstack((image_pts_x, image_pts_y)).reshape(-1,7,2)
        # test:可视化
        #from visualize.visualize import visualize_point_line_image
        #image_pts = image_pts.reshape(-1, 2)
        #image_pts = np.clip(image_pts,[0, 0],[h-1, w-1])
        #visualize_point_line_image(img, np.array([[]]), image_pts)
                
        ## 4.循环不同点的组合，获取组合的support point和线段长度，从候选线段中确定结果
        image_lines = select_from_candidate_lines(image_pts, valid_pts_mask)

        ## 5.Handle border
        new_lines, valid = clip_line_to_boundaries(image_lines, img.shape, min_len=10)
        image_lines = new_lines[valid]         
        
        return image_lines    


def views_transform(img, nimage_lines, cam_dict, nimg, ndepth, ncam_dict):
    '''
    邻近视角投影到当前视角，用深度突变边界剔除
    '''
    # 如果为当前视角，那就无需投影，直接返回lines
    if cam_dict['img_name'] == ncam_dict['img_name']:
        return nimage_lines
    ## 1.在线段中插入5个点
    nimage_pts = interpolate_line_segments(nimage_lines, num_points=5)
    # test:可视化
    #from visualize.visualize import visualize_point_line_image
    #visualize_point_line_image(nimg, nimage_lines, nimage_pts)

    ## 2.赋值深度
    ndepth_scale = ndepth.shape[0] / nimg.shape[0]
    nimage_pts_depth = get_depth_from_depthmap(ndepth, nimage_pts, scale=ndepth_scale, search_window=5)
    # 剔除深度为有效值(不为0)的点数少于2的线段
    nimage_pts_depth_ = nimage_pts_depth > 0
    valid_line_mask = np.sum(nimage_pts_depth_, axis=1) > 1
    nimage_pts_depth = nimage_pts_depth[valid_line_mask]
    nimage_pts = nimage_pts[valid_line_mask]
    # 仍有深度为0的点，为了保持每条线段有7个点的条件
    valid_pts_mask = nimage_pts_depth > 0

    ## 3.投影
    # 投影到三维中
    nimage_pts_x = nimage_pts[:, :, 0]  # 行号
    nimage_pts_y = nimage_pts[:, :, 1]  # 列号  
    homo_pts3D = cam2world(nimage_pts_x, nimage_pts_y, nimage_pts_depth, ncam_dict)
    # test:可视化
    #viz_points2(homo_pts3D[:3, :].T)
    # 投影到当前视角中
    image_pts_x, image_pts_y, _= world2cam(homo_pts3D, cam_dict)
    image_pts = np.hstack((image_pts_x, image_pts_y)).reshape(-1,7,2)
    # test:可视化
    #from visualize.visualize import visualize_point_line_image
    #image_pts = image_pts.reshape(-1, 2)
    #image_pts = np.clip(image_pts,[0, 0],[h-1, w-1])
    #visualize_point_line_image(img, np.array([[]]), image_pts)
            
    ## 4.循环不同点的组合，获取组合的support point和线段长度，从候选线段中确定结果
    image_lines = select_from_candidate_lines(image_pts, valid_pts_mask)

    ## 5.Handle border
    new_lines, valid = clip_line_to_boundaries(image_lines, img.shape, min_len=10)
    image_lines = new_lines[valid]         
    
    return image_lines    


def simplify_and_fill_mask(count):
    """
    将带有空洞的二值掩码转换为一个填充完整的、覆盖所有原始区域的最小凸多边形。
    
    Args:
        count (np.ndarray): 原始的二值掩码 (0/1)。
        
    Returns:
        np.ndarray: 经过凸包填充后的新二值掩码 (0/1)。
    """
    # 1. 查找所有有效像素的坐标 (这是关键的第一步)
    # np.where(count > 0) 返回 (行索引数组, 列索引数组)
    rows, cols = np.where(count > 0)
    
    # 组织成 N x 2 的点集 (x, y)，即 (列, 行)
    points = np.vstack((cols, rows)).T  

    # 2. 确保点集不为空
    if points.shape[0] == 0:
        # 如果没有有效点，返回全零掩码
        return np.zeros_like(count, dtype=np.uint8)
        
    # 3. 计算整个点集的凸包
    # hull 包含覆盖所有点的最小凸多边形的顶点坐标
    hull = cv2.convexHull(points, returnPoints=True)

    # 4. 创建新的空白掩码
    # 必须确保 dtype 为 np.uint8，以便 cv2.drawContours 正常工作
    filled_mask = np.zeros_like(count, dtype=np.uint8)

    # 5. 用这个单一的凸包轮廓进行填充
    cv2.drawContours(
        filled_mask, 
        [hull],        # 传入包含唯一凸包的列表
        -1,            # 绘制所有轮廓 (在这里就是第 0 个)
        color=1,       # 填充颜色为 1
        thickness=cv2.FILLED # 填充整个区域
    )

    return filled_mask

def grid_reprojection(nimg, ndepth, ncam_dict, img, depth, cam_dict, H):
    '''
    将渲染的新视角图像各像素投影到当前视角下，获得两视角间的重叠区域
        - Args:
            - ndepth:新视角图像的深度图
            - ncam_dict:新视角的内外参字典
            - depth:当前视角的深度图
            - cam_dict:当前视角的内外参字典
            - scale:缩小的倍数
        - Returns:    
            - count:在当前视角图像中的重叠区域，用0和1区分
    '''
    ############################## 重采样  ##############################
    # 1.在当前视角下初始化格网坐标
    height = img.shape[0]
    width = img.shape[1]
    grid = np.indices((height, width)).reshape(2, -1)  # shape[2, h, w]-->shape[2, h*w]
    # 如果为当前视角，那就无需投影
    if cam_dict['img_name'] == ncam_dict['img_name']:    
        nx = grid[0].reshape(-1, 1)
        ny = grid[1].reshape(-1, 1)
    else:
        # 2.获取深度
        scale_x = depth.shape[0] / img.shape[0]
        scale_y = depth.shape[1] / img.shape[1]
        grid_depth = bilinear_interpolate(depth, grid[0], grid[1], [scale_x, scale_y])  # grid[0]为行号，grid[1]为列号
        # 3.坐标转换：从新视角中取值，即从当前视角投影到新视角
        homo_pts3D = cam2world(grid[0], grid[1], grid_depth, cam_dict)
        nx, ny, _= world2cam(homo_pts3D, ncam_dict)

    # 4.坐标转换：从新视角转换到经过H变换后的
    warped_pts = warp_points(np.hstack((nx, ny)), H)
    # 5.reshape，方便生成mask
    nx = nx.reshape((height, width))  # nx为从上到下
    ny = ny.reshape((height, width))  # ny为从左到右
    nx_H = warped_pts[:, 0].reshape((height, width))
    ny_H = warped_pts[:, 1].reshape((height, width))
    ######################### 获取重叠区域  ##############################
    # mask1：投影后在新视角航片x坐标范围内的像素
    mask1 = (nx >= 0) & (nx <= (nimg.shape[0]))
    # mask2:投影后在新视角航片y坐标范围内的像素
    mask2 = (ny >= 0) & (ny <= (nimg.shape[1]))
    # mask3:进一步做H变换后，x坐标在范围内
    mask3 = (nx_H >= 0) & (nx_H <= (nimg.shape[0]))
    # mask4:进一步做H变换后，y坐标在范围内
    mask4 = (ny_H >= 0) & (ny_H <= (nimg.shape[1]))    
    # 同时满足四个条件
    mask = mask1 & mask2 & mask3 & mask4
    # 确保 count 是 uint8 类型，值域为 0 和 1
    count = np.where(mask, 1, 0).astype(np.uint8) 

    ######################### 轮廓检测与内部填充 ##########################
    count = simplify_and_fill_mask(count)

    return count



def views_transform(img, nimage_lines, cam_dict, nimg, ndepth, ncam_dict):
    '''
    邻近视角投影到当前视角，用深度突变边界剔除
    '''
    # 如果为当前视角，那就无需投影，直接返回lines
    if cam_dict['img_name'] == ncam_dict['img_name']:
        return nimage_lines
    ## 1.在线段中插入5个点
    nimage_pts = interpolate_line_segments(nimage_lines, num_points=5)
    # test:可视化
    #from visualize.visualize import visualize_point_line_image
    #visualize_point_line_image(nimg, nimage_lines, nimage_pts)

    ## 2.赋值深度
    ndepth_scale = ndepth.shape[0] / nimg.shape[0]
    nimage_pts_depth = get_depth_from_depthmap(ndepth, nimage_pts, scale=ndepth_scale, search_window=5)
    # 剔除深度为有效值(不为0)的点数少于2的线段
    nimage_pts_depth_ = nimage_pts_depth > 0
    valid_line_mask = np.sum(nimage_pts_depth_, axis=1) > 1
    nimage_pts_depth = nimage_pts_depth[valid_line_mask]
    nimage_pts = nimage_pts[valid_line_mask]
    # 仍有深度为0的点，为了保持每条线段有7个点的条件
    valid_pts_mask = nimage_pts_depth > 0

    ## 3.投影
    # 投影到三维中
    nimage_pts_x = nimage_pts[:, :, 0]  # 行号
    nimage_pts_y = nimage_pts[:, :, 1]  # 列号  
    homo_pts3D = cam2world(nimage_pts_x, nimage_pts_y, nimage_pts_depth, ncam_dict)
    # test:可视化
    #viz_points2(homo_pts3D[:3, :].T)
    # 投影到当前视角中
    image_pts_x, image_pts_y, _= world2cam(homo_pts3D, cam_dict)
    image_pts = np.hstack((image_pts_x, image_pts_y)).reshape(-1,7,2)
    # test:可视化
    #from visualize.visualize import visualize_point_line_image
    #image_pts = image_pts.reshape(-1, 2)
    #image_pts = np.clip(image_pts,[0, 0],[h-1, w-1])
    #visualize_point_line_image(img, np.array([[]]), image_pts)
            
    ## 4.循环不同点的组合，获取组合的support point和线段长度，从候选线段中确定结果
    image_lines = select_from_candidate_lines(image_pts, valid_pts_mask)

    ## 5.Handle border
    new_lines, valid = clip_line_to_boundaries(image_lines, img.shape, min_len=10)
    image_lines = new_lines[valid]         
    
    return image_lines    

