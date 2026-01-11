import os
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict

def parse_line_segments(line3dpp_folder, img_id, width, height):
    """
    解析XML文件中的线段数据，并转换为指定格式的numpy数组。

    参数:
        line3dpp_folder: line3dpp重建结果的文件夹路径
        img_id: 图像ID
        width: 图像宽度
        height: 图像高度

    返回:
        segments: 包含线段信息的numpy数组，形状为(N, 4)，其中N为线段数量。
    """
    filename = 'segments_L3D++_'+ str(img_id) + '_' + str(width) + 'x' + str(height) + '.bin'
    filepath = os.path.join(line3dpp_folder, 'L3D++_data', filename)
    # 解析XML文件
    tree = ET.parse(filepath)
    root = tree.getroot()

    # 找到所有 <item> 元素
    items = root.findall(".//item")

    segments = []
    for item in items:
        x = float(item.find("x").text)
        y = float(item.find("y").text)
        z = float(item.find("z").text)
        w = float(item.find("w").text)
        segments.append([y, x, w, z])
    
    # 将列表转换为numpy数组
    segments = np.array(segments, dtype=np.float32) 
    # 端点坐标调整到范围内
    segments[:, 0] = np.clip(segments[:, 0], 0, height - 1)
    segments[:, 1] = np.clip(segments[:, 1], 0, width - 1)
    segments[:, 2] = np.clip(segments[:, 2], 0, height - 1)
    segments[:, 3] = np.clip(segments[:, 3], 0, width - 1)

    return segments  # (N, 4)


def parse_lines3dpp(folder_path):
    """
    解析 txt 文件，将 3D 线段存储到矩阵中，将对应的 2D 线段存储到列表中。

    参数:
    - folder_path: 包含 3D 线段数据的文件夹路径。

    返回:
    - lines3d: numpy 数组，存储所有 3D 线段，形状为 (total_3d_lines, 6)
    - lines3d_to_lines2d: 字典，{line3d_id: [(cam_id, seg_id)..]}
    - lines2d_in_cam: 字典，{cam_id：{seg_id:{'coord': (x1,y1,x2,y2), 'mask_id': -1}}}
    """
    lines3d = []  # 用于存储所有 3D 线段
    residuals2d_for_lines3d = []  # 存储每个 3D 线段对应的 2D 线段信息
    lines3d_to_lines2d = {}  # {line3d_id: [(cam_id, seg_id)..]}
    lines2d_in_cam = defaultdict(lambda: defaultdict(dict)) # {cam_id：{seg_id:{'coord': (x1,y1,x2,y2), 'mask_id': -1}}}

    # 在指定文件夹中查找符合条件的文件
    for filename in os.listdir(folder_path):
        if filename.startswith("Line3D++") and filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            break
    
    with open(file_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            
            # 解析 3D 线段
            n = int(data[0])  # 3D 线段数量
            line3d = np.array(data[1:1 + n * 6], dtype=float).reshape(n, 6)
            lines3d.append(line3d)
            
            # 解析 2D 线段
            m_start_idx = 1 + n * 6
            m = int(data[m_start_idx])  # 2D 线段数量
            residuals2d = []
            for i in range(m):
                start_idx = m_start_idx + 1 + i * 6
                cam_id = int(data[start_idx])
                seg_id = int(data[start_idx + 1])
                p1x, p1y, q1x, q1y = map(float, data[start_idx + 2:start_idx + 6])
                residuals2d.append((cam_id, seg_id, p1x, p1y, q1x, q1y))
            for j in range(n):
                if n > 1:
                    a = 0
                residuals2d_for_lines3d.append(residuals2d)

    for line3d_id, residuals2d in enumerate(residuals2d_for_lines3d):
        lines2d_id = []
        for line2d in residuals2d:
            cam_id = line2d[0]
            seg_id = line2d[1]
            lines2d_id.append((cam_id, seg_id))
            lines2d_in_cam[cam_id][seg_id] = {
                'coord': line2d[2:],
                'mask': -1
            }
        lines3d_to_lines2d[line3d_id] = lines2d_id
    
    # 将所有 3D 线段合并为一个矩阵
    lines3d = np.vstack(lines3d).reshape(-1, 6)

    return lines3d, lines3d_to_lines2d, lines2d_in_cam


def save_segments_l3dpp(lines, output_folder, img_name, width, height):
    """
    将线段以 Boost-Serialization 风格保存为 XML 文件 (.bin)
    lines: (N, 4) [x1, y1, x2, y2]
    """
    import xml.dom.minidom as minidom

    filename = f'segments_L3D++_{img_name}_{width}x{height}.bin'
    l3dpp_dir = os.path.join(output_folder, 'L3D++_data')
    os.makedirs(l3dpp_dir, exist_ok=True)
    filepath = os.path.join(l3dpp_dir, filename)

    num_lines = len(lines)

    # 创建 XML 结构
    root = ET.Element('boost_serialization', {
        'signature': 'serialization::archive',
        'version': '18'
    })
    data = ET.SubElement(root, 'data', {
        'class_id': '0',
        'tracking_level': '0',
        'version': '0'
    })

    ET.SubElement(data, 'width_').text = str(num_lines)
    ET.SubElement(data, 'height_').text = '1'
    ET.SubElement(data, 'real_width_').text = str(num_lines)
    ET.SubElement(data, 'pitchCPU_').text = str(num_lines * 16)
    ET.SubElement(data, 'strideCPU_').text = str(num_lines)
    ET.SubElement(data, 'pitchGPU_').text = '0'
    ET.SubElement(data, 'strideGPU_').text = '0'

    for i, line in enumerate(lines):
        if i == 0:
            item = ET.SubElement(data, 'item', {
                'class_id': '1',
                'tracking_level': '0',
                'version': '0'
            })
        else:
            item = ET.SubElement(data, 'item')
        ET.SubElement(item, 'x').text = f"{line[0]:.9e}"
        ET.SubElement(item, 'y').text = f"{line[1]:.9e}"
        ET.SubElement(item, 'z').text = f"{line[2]:.9e}"
        ET.SubElement(item, 'w').text = f"{line[3]:.9e}"

    # 使用 minidom 格式化（不写入声明）
    xml_str = ET.tostring(root, encoding='utf-8')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="\t", encoding='utf-8').decode('utf-8')

    # 去掉 minidom 自动添加的 XML 声明
    pretty_xml = "\n".join(line for line in pretty_xml.splitlines() if not line.strip().startswith("<?xml"))

    # 去掉最外层的一个制表符层级
    fixed_lines = []
    for line in pretty_xml.splitlines():
        if line.startswith("\t"):
            fixed_lines.append(line[1:])  # 删除一个制表符
        else:
            fixed_lines.append(line)
    pretty_xml = "\n".join(fixed_lines)

    # 写入文件（仅保留我们定义的头部）
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>\n')
        f.write('<!DOCTYPE boost_serialization>\n')
        f.write(pretty_xml)

    return filepath


def visualize_lines2d(lines2d_in_cam, image_shape, cam_id):
    """
    可视化指定相机中的 2D 线段。

    参数:
    - lines2d_in_cam: 字典，包含每个相机的 2D 线段信息
    - image_shape: 图像的形状 (高度, 宽度)
    - cam_id: 要可视化的相机ID
    """
    import matplotlib.pyplot as plt
    
    if cam_id not in lines2d_in_cam:
        print(f"相机ID {cam_id} 不存在。")
        return

    lines = lines2d_in_cam[cam_id]

    for line2d in lines2d_in_cam[cam_id].values():
        p1x, p1y, q1x, q1y = line2d['coord']
        plt.plot([p1x, q1x], [p1y, q1y], 'b-')

    plt.xlim(0, image_shape[1])
    plt.ylim(image_shape[0], 0)
    plt.title(f'2D Lines in Camera {cam_id}')
    plt.show()


if __name__ == '__main__':
    lines3d, lines3d_to_lines2d, lines2d_in_cam = parse_lines3dpp('/home/rylynn/Pictures/Clustering_Workspace/Shanghai_Region5/Line3D++')
    visualize_lines2d(lines2d_in_cam, (1080, 1920), 0)