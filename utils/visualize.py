import argparse
import numpy as np
import open3d
import os

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