import argparse
import numpy as np
import open3d
import os
import cv2

import matplotlib
import matplotlib.pyplot as plt
from deeplsd.geometry.viz_2d import get_flow_vis, plot_images, plot_lines, save_plot


class Model:
    def __init__(self):
        self.__vis = None


    def draw_lines(self, lines3D, color=[0, 0, 0]):
        lines_num = lines3D.shape[0]
        # 1.lines3D: (N,2,3)->(2N,3)
        lines3D = lines3D.reshape(-1, 3)
        # 2.线段两个端点对应的行号
        pairs = np.arange(lines3D.shape[0]).reshape(-1, 2)
        # 3.设定线段颜色
        colors = [color for i in range(lines_num)]
        # 4.绘制线段
        line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(lines3D),
        lines=open3d.utility.Vector2iVector(pairs),
        )
        line_set.colors = open3d.utility.Vector3dVector(colors)
        self.__vis.add_geometry(line_set)

    def create_window(self):
        self.__vis = open3d.visualization.Visualizer()
        self.__vis.create_window()

    def show(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()
        self.__vis.destroy_window()

def viz_lines3D(lines3D):
    # read COLMAP model
    model = Model()

    # display using Open3D visualization tools
    model.create_window()
    model.draw_lines(lines3D, color=[0.8, 0.3, 0.3])
    model.show()    


def viz_lines2D(img, lines, output_path, img_name, ncam_id, name):
    '''
    
    '''
    plot_images([img], dpi=100, size=40)
    plot_lines([lines[:,:, [1, 0]]], indices=range(1))
    path = os.path.join(output_path, img_name)
    os.makedirs(path, exist_ok=True)
    save_plot(os.path.join(path, '{0:05d}'.format(ncam_id) + '_' + name + '.jpg'))


def viz_lines2D2(img, lines, output_path, name):
    '''
    
    '''
    plot_images([img], dpi=100, size=40)
    plot_lines([lines[:,:, [1, 0]]], indices=range(1))
    save_plot(os.path.join(output_path, name + '.jpg'))
    plt.close()


def viz_points(points):
    """
    可视化三维点云.
    points:形状为[n, 3]
    """
    # 确保输入是 numpy 数组
    if not isinstance(points, np.ndarray):
        raise ValueError("输入必须是 numpy 数组")
   
    # 转换为 open3d 的点云格式
    pcd = open3d.geometry.PointCloud()

    pcd.points = open3d.utility.Vector3dVector(points)

    pcd.colors = open3d.utility.Vector
    
    # 可视化点云
    open3d.visualization.draw_geometries([pcd])


def viz_points2(points):
    """
    可视化三维点云.
    points:形状为[n, 3]
    """
    # 确保输入是 numpy 数组
    if not isinstance(points, np.ndarray):
        raise ValueError("输入必须是 numpy 数组")

    # 转换为 open3d 的点云格式
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)

    # 设置所有点的颜色为红色（RGB 值，范围从 0 到 1）
    color = [0, 0, 0]  # 红色
    num_points = len(points)
    colors = np.tile(color, (num_points, 1))
    pcd.colors = open3d.utility.Vector3dVector(colors)

    # 创建可视化窗口
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # 获取渲染选项并设置点的大小
    opt = vis.get_render_option()
    opt.point_size = 0.5  # 设置点的大小，可根据需要调整

    # 运行可视化
    vis.run()


def visualize_points_with_depth_color(points, depths, cmap='viridis_r'):
    """
    根据深度值为点云赋色并进行可视化.

    参数:
    points (numpy.ndarray): 形状为 (3, N) 的点坐标矩阵，其中每一列是一个点的 (x, y, z) 坐标.
    depths (numpy.ndarray): 形状为 (N,) 的深度值数组.
    cmap (str): matplotlib 的色图名称，默认为 'viridis'.
    """
    # 创建点云对象
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points.T)  # 转置以匹配 open3d 的格式

    # 将深度值映射到颜色
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap)
    normal = (depths - np.min(depths)) / (np.max(depths) - np.min(depths))
    print(np.max(normal))
    print(np.min(normal))
    colors = cmap((depths - np.min(depths)) / (np.max(depths) - np.min(depths)))[:, :3]  # 归一化并映射到 RGB

    # 设置点云颜色
    pcd.colors = open3d.utility.Vector3dVector(colors)

    # 可视化点云
    open3d.visualization.draw_geometries([pcd])


def visualize_point_line_image(img, lines, points):
    '''
    - 输入
        - lines:(N,4,2), 如果为空则不绘制线段
        - points:任意形状,都会被reshape为(N, 2)
    '''
    # 绘制图像和点
    plt.imshow(img, cmap='gray')  # 显示图像
    if lines.size != 0:
        # 在图像上绘制线段
        for line in lines:
            start_point = line[0, [1, 0]]  # 线段起点
            end_point = line[1, [1, 0]]    # 线段终点
            plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                    color='blue', linewidth=2, label='Lines')  # 绘制线段
    # 在图像上绘制点
    points = points.reshape(-1, 2)
    plt.scatter(points[:, 1], points[:, 0], color='red', s=4, zorder=5)  # s 是点的大小

    # 设置坐标轴（可选）
    plt.axis('off')  # 关闭坐标轴
    plt.title("Image with Points")  # 添加标题
    # 显示图像
    plt.show()


def viz_pairwise_matches(img1, lines1, img2, lines2, matches, output_path, name_prefix):
    """
    可视化两张图片之间的线段匹配。
    布局：左边为邻域图像 (img2)，右边为原图像 (img1)
    """
    # 1. 图像预处理
    img1_c = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if len(img1.shape) == 2 else img1.copy()
    img2_c = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if len(img2.shape) == 2 else img2.copy()

    h1, w1 = img1_c.shape[:2]
    h2, w2 = img2_c.shape[:2]

    # 2. 创建拼接画布 (邻域图在左 w2, 原图在右 w1)
    vis_h = max(h1, h2)
    vis_w = w1 + w2
    vis_img = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
    
    # --- 修改位置：先填入 img2 (左)，再填入 img1 (右) ---
    vis_img[:h2, :w2] = img2_c
    vis_img[:h1, w2:w2+w1] = img1_c

    # 3. 绘制匹配线段
    np.random.seed(42) 
    
    for (idx1, idx2) in matches:
        color = np.random.randint(0, 255, 3).tolist()
        
        # --- 左图：邻域图像 (img2, lines2) ---
        # 直接使用原始坐标，无需偏移
        l2 = lines2[idx2]
        pt2_a = (int(l2[0][1]), int(l2[0][0])) 
        pt2_b = (int(l2[1][1]), int(l2[1][0]))
        
        # --- 右图：原图像 (img1, lines1) ---
        # 需要加上左图 (img2) 的宽度 w2 作为偏移量
        l1 = lines1[idx1]
        pt1_a = (int(l1[0][1]) + w2, int(l1[0][0]))
        pt1_b = (int(l1[1][1]) + w2, int(l1[1][0]))

        # 执行绘制
        cv2.line(vis_img, pt2_a, pt2_b, color, 16, cv2.LINE_AA) # 在左图画线
        cv2.line(vis_img, pt1_a, pt1_b, color, 16, cv2.LINE_AA) # 在右图画线
        
        # (可选) 绘制跨图像的连接线，增强视觉直观性
        center2 = ((pt2_a[0]+pt2_b[0])//2, (pt2_a[1]+pt2_b[1])//2)
        center1 = ((pt1_a[0]+pt1_b[0])//2, (pt1_a[1]+pt1_b[1])//2)
        cv2.line(vis_img, center2, center1, color, 2, cv2.LINE_AA)

    # 4. 保存结果
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    save_name = os.path.join(output_path, f"{name_prefix}_swapped.jpg")
    cv2.imwrite(save_name, vis_img)


def viz_pairwise_matches_original(img1, lines1, img2, lines2, matches, output_path, name_prefix):
    """
    可视化两张图片之间的线段匹配。
    
    参数:
        img1: 源图片 (灰度或彩色)
        lines1: 源图片的所有线段，形状 (N, 2, 2)，格式 [[y1, x1], [y2, x2]]
        img2: 邻域图片
        lines2: 邻域图片的所有线段
        matches: 匹配列表，每一项为 (idx_in_lines1, idx_in_lines2)
        output_path: 保存路径
        name_prefix: 保存文件名前缀
    """
    # 1. 图像预处理：转换为彩色以便绘制彩色线段
    if len(img1.shape) == 2:
        img1_c = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_c = img1.copy()
        
    if len(img2.shape) == 2:
        img2_c = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2_c = img2.copy()

    h1, w1 = img1_c.shape[:2]
    h2, w2 = img2_c.shape[:2]

    # 2. 创建拼接画布 (高度取最大值，宽度相加)
    vis_h = max(h1, h2)
    vis_w = w1 + w2
    vis_img = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
    
    # 填入图片
    vis_img[:h1, :w1] = img1_c
    vis_img[:h2, w1:w1+w2] = img2_c

    # 3. 绘制匹配线段
    # 设置随机种子以保证颜色一致性
    np.random.seed(42) 
    
    for (idx1, idx2) in matches:
        # 生成随机颜色 (B, G, R)
        color = np.random.randint(0, 255, 3).tolist()
        
        # 获取源线段 (注意：你的数据格式是 [[y, x], [y, x]])
        # OpenCV画线需要 (x, y)
        l1 = lines1[idx1]
        pt1_a = (int(l1[0][1]), int(l1[0][0])) 
        pt1_b = (int(l1[1][1]), int(l1[1][0]))
        
        # 获取目标线段 (并加上左图宽度的偏移量)
        l2 = lines2[idx2]
        pt2_a = (int(l2[0][1]) + w1, int(l2[0][0]))
        pt2_b = (int(l2[1][1]) + w1, int(l2[1][0]))

        # 在左图画线
        cv2.line(vis_img, pt1_a, pt1_b, color, 16, cv2.LINE_AA)
        # 在右图画线
        cv2.line(vis_img, pt2_a, pt2_b, color, 16, cv2.LINE_AA)
        
        # (可选) 如果你想画一条连线连接两个线段的中心，可以取消下面注释
        # center1 = ((pt1_a[0]+pt1_b[0])//2, (pt1_a[1]+pt1_b[1])//2)
        # center2 = ((pt2_a[0]+pt2_b[0])//2, (pt2_a[1]+pt2_b[1])//2)
        # cv2.line(vis_img, center1, center2, color, 1, cv2.LINE_AA)

    # 4. 保存结果
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    save_name = os.path.join(output_path, f"{name_prefix}.jpg")
    cv2.imwrite(save_name, vis_img)