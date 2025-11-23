import numpy as np
import open3d as o3d

def parse_obj_file(file_path):
    vertices = []
    texture_coords = []
    normals = []
    faces = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  # 顶点
                vertex = list(map(float, line.strip().split()[1:4]))
                vertices.append(vertex)
            elif line.startswith('vt '):  # 纹理坐标
                tex_coord = list(map(float, line.strip().split()[1:3]))
                texture_coords.append(tex_coord)
            elif line.startswith('vn '):  # 法线
                normal = list(map(float, line.strip().split()[1:4]))
                normals.append(normal)
            elif line.startswith('f '):  # 面
                face_data = line.strip().split()[1:]
                face = []
                for vertex_data in face_data:
                    indices = vertex_data.split('/')
                    vertex_idx = int(indices[0]) - 1
                    
                    tex_idx = None
                    if len(indices) > 1 and indices[1]:
                        tex_idx = int(indices[1]) - 1
                    
                    normal_idx = None
                    if len(indices) > 2 and indices[2]:
                        normal_idx = int(indices[2]) - 1
                    
                    face.append((vertex_idx, tex_idx, normal_idx))
                faces.append(face)
    
    return {
        'vertices': vertices,
        'texture_coords': texture_coords,
        'normals': normals,
        'faces': faces
    }

def create_line_segments_from_vertices(vertices, method='pairs'):
    """
    将顶点两两组合为线段
    
    参数:
    vertices: 顶点列表
    method: 组合方式
        'sequential' - 顺序连接 (v0-v1, v1-v2, v2-v3, ...)
        'pairs' - 成对连接 (v0-v1, v2-v3, v4-v5, ...)
        'all_edges' - 基于面的边连接
    """
    vertices = np.array(vertices)
    line_segments = []
    
    if method == 'sequential':
        # 顺序连接相邻顶点
        for i in range(len(vertices) - 1):
            line_segment = np.concatenate([vertices[i], vertices[i + 1]])
            line_segments.append(line_segment)
    
    elif method == 'pairs':
        # 成对连接顶点
        for i in range(0, len(vertices) - 1, 2):
            line_segment = np.concatenate([vertices[i], vertices[i + 1]])
            line_segments.append(line_segment)
    
    elif method == 'all_edges':
        # 基于面的边连接（需要faces信息）
        pass
    
    return np.array(line_segments)

def create_line_segments_from_faces(vertices, faces):
    """
    基于面的边创建线段，避免重复边
    """
    vertices = np.array(vertices)
    edges_set = set()
    line_segments = []
    
    for face in faces:
        face_vertices = [vertex[0] for vertex in face]  # 只取顶点索引
        
        # 为面的每个边创建线段
        for i in range(len(face_vertices)):
            start_idx = face_vertices[i]
            end_idx = face_vertices[(i + 1) % len(face_vertices)]
            
            # 确保边的唯一性（无序对）
            edge = tuple(sorted([start_idx, end_idx]))
            
            if edge not in edges_set:
                edges_set.add(edge)
                line_segment = np.concatenate([vertices[start_idx], vertices[end_idx]])
                line_segments.append(line_segment)
    
    return np.array(line_segments)

def visualize_line_segments(line_segments, vertices=None):
    """
    使用Open3D可视化线段
    """
    # 创建LineSet对象
    line_set = o3d.geometry.LineSet()
    
    # 提取所有唯一的点
    all_points = []
    line_indices = []
    point_index_map = {}
    
    for i, segment in enumerate(line_segments):
        start_point = segment[:3]
        end_point = segment[3:]
        
        # 检查点是否已存在
        start_key = tuple(start_point)
        end_key = tuple(end_point)
        
        if start_key not in point_index_map:
            point_index_map[start_key] = len(all_points)
            all_points.append(start_point)
        
        if end_key not in point_index_map:
            point_index_map[end_key] = len(all_points)
            all_points.append(end_point)
        
        line_indices.append([point_index_map[start_key], point_index_map[end_key]])
    
    # 设置点和线
    line_set.points = o3d.utility.Vector3dVector(np.array(all_points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(line_indices))
    
    # 设置线段颜色（红色）
    colors = [[1, 0, 0] for _ in range(len(line_indices))]  # 红色线段
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # 如果提供了原始顶点，也显示出来
    geometries = [line_set]
    
    if vertices is not None:
        # 创建点云显示顶点
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(vertices))
        point_cloud.paint_uniform_color([0, 1, 0])  # 绿色顶点
        geometries.append(point_cloud)
    
    # 可视化
    o3d.visualization.draw_geometries(geometries, window_name="OBJ线段可视化")

# 主函数
def main():
    # 读取OBJ文件
    obj_file_path = "/home/rylynn/Documents/CloudCompare_Exports/entity_04.obj"  # 替换为你的OBJ文件路径
    obj_data = parse_obj_file(obj_file_path)
    
    vertices = obj_data['vertices']
    faces = obj_data['faces']
    
    print(f"读取到 {len(vertices)} 个顶点，{len(faces)} 个面")
    
    # 方法1: 基于面的边创建线段（推荐）
    if len(faces) > 0:
        print("使用基于面的边创建线段...")
        line_segments = create_line_segments_from_faces(vertices, faces)
    else:
        print("使用顺序连接创建线段...")
        line_segments = create_line_segments_from_vertices(vertices, method='pairs')
    
    print(f"创建了 {len(line_segments)} 条线段")
    
    # 可视化线段
    visualize_line_segments(line_segments, vertices)

if __name__ == "__main__":
    # 创建测试文件（如果没有OBJ文件的话）
    # create_sample_obj_file()
    
    main()