import os
import sys
import numpy as np
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from datasets.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary, read_extrinsics_text, read_intrinsics_text, \
 read_points3D_binary, read_points3D_text


def readColmapCameras(cam_extrinsics, cam_intrinsics, image_scale):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = (intr.height) / image_scale
        width = (intr.width) / image_scale

        uid = extr.id
        image_name = (extr.name).split('.')[0]
        R = np.transpose(qvec2rotmat(extr.qvec)) # colmap中的R和T：世界坐标转到相机坐标
        T = (-R @ np.array(extr.tvec)) / image_scale

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0] / image_scale
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0] / image_scale
            focal_length_y = intr.params[1] / image_scale
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # 去掉没有参与匹配的特征点
        match_point = np.where(extr.point3D_ids != -1)
        xys = extr.xys[match_point]
        point3D_ids = extr.point3D_ids[match_point]
        # 将point3D和point2D形成对应关系
        points3D_to_xys = dict(zip(point3D_ids, xys))

        cam_infos.append({
            "id": uid,
            "img_name": image_name,
            "width": width,
            "height": height,
            "position": T,
            "rotation": R,
            "fx": focal_length_x,
            "fy": focal_length_y,
            "points3D_ids": point3D_ids,
            "points3D_to_xys": points3D_to_xys
        })

    sys.stdout.write('\n')
    return cam_infos

def load_sparse_model(path_to_model, image_scale):
    '''
    获取相片的内外参,加载3D points和相片之间的关系
    - args
        - path_to_model:Colmap稀疏重建结果的路径
        - image_scale:相片缩小的倍数
    - return
        - camera_dict:字典,和GS的cameras.json相同
        - points_in_images:字典
    '''
    camerasInfo = {}
    points3d = {}

    # 读取文件
    try:
        cameras_extrinsic_file = os.path.join(path_to_model, "images.bin")
        cameras_intrinsic_file = os.path.join(path_to_model, "cameras.bin")
        points3d_file = os.path.join(path_to_model, "points3D.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        points3d = read_points3D_binary(points3d_file)
    except:
        cameras_extrinsic_file = os.path.join(path_to_model, "images.txt")
        cameras_intrinsic_file = os.path.join(path_to_model, "cameras.txt")
        points3d_file = os.path.join(path_to_model, "points3D.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        points3d = read_points3D_text(points3d_file)

    # 获取相机内外参
    camerasInfo_unsorted = readColmapCameras(cam_extrinsics, cam_intrinsics, image_scale)
    camerasInfo = sorted(camerasInfo_unsorted, key=lambda x: x["id"])

    # 如果起始相机ID是1，则将其更新为0
    if camerasInfo[0]['id'] == 1:
        camerasInfo = [{**item, 'id': item['id'] - 1} for item in camerasInfo]
        points_in_images = []
        # 加载3D sparse point时也需要注意更新为0
        for key in points3d.keys():
            points_in_images.append(points3d[key].image_ids - 1)
        return camerasInfo, points_in_images
    # 起始ID是0就无需更新
    else:
        for key in points3d.keys():
            points_in_images.append(points3d[key].image_ids - 1)
        return camerasInfo, points_in_images


def match_pair(camerasInfo, points_in_images, match_point_num=0):
    '''
    统计相片之间公共特征点的数量，根据阈值构建重叠关系
    '''
    num_cameras = len(camerasInfo)
    matches_matrix = np.zeros((num_cameras, num_cameras), dtype=int)
    for camera_ids in points_in_images:
        for i in range(len(camera_ids)):
            for j in range(i, len(camera_ids)):
                matches_matrix[camera_ids[i], camera_ids[j]] += 1 
                matches_matrix[camera_ids[j], camera_ids[i]] += 1
    overlap_images = {}
    for i, matches in enumerate(matches_matrix):
        overlap = np.where(matches > match_point_num)[0].tolist()
        overlap_images[i] = overlap

    matches_matrix = np.argsort(-matches_matrix, axis=1)
    return matches_matrix, overlap_images


def find_common_points(img1_id, img2_id, camerasInfo):
    # 取交集
    common_point_ids = np.intersect1d(camerasInfo[img1_id]['points3D_ids'], camerasInfo[img2_id]['points3D_ids'])

    common_points = []
    for point_id in common_point_ids:
        pt1 = camerasInfo[img1_id]['points3D_to_xys'][point_id]
        pt2 = camerasInfo[img2_id]['points3D_to_xys'][point_id]
        common_points.append((pt1, pt2))
    return np.array(common_points)


def compute_bounding_box(points):
    '''计算2D点的最小外接矩形'''
    if len(points) == 0:
        return 0
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    area = (x_max - x_min) * (y_max - y_min)
    return area


def read_depth(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":  # b是byte的意思
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    depth_map = np.transpose(array, (1, 0, 2)).squeeze()
    return depth_map


def read_cam_dict(cam_dict):
    '''
    从cam_dict字典中读取相机内外参信息
    '''
    pos = np.array(cam_dict['position'])
    rot = np.array(cam_dict['rotation'])
    return pos, rot, cam_dict['fx'], cam_dict['fy'], cam_dict['width'], cam_dict['height']


def load_depth_float32(depth_path):
    # 1. 读取数据
    # 使用 UNCHANGED 或 ANYDEPTH 确保读取原始的 float32 数据
    image = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0] #(H, W)
    

    # 2. 处理无效值 (Sky/Infinity)
    # 在 float32 模式下，天空可能不再是 65504，而是非常大的数或 np.inf
    # 你可以根据需要设置一个阈值，比如 1000米 或 100000厘米
    # 这里示例将其标记为无效（如果需要掩码）
    # invalid_mask = (image > 1e9) | np.isinf(image)

    # 3. 单位转换：获取 COLMAP 定义的深度值
    # 文档明确指出："raw data ... is cm" (原始数据是厘米)
    # COLMAP 标准输出通常使用 "米" (Meters)
    # 提供的参考代码用了 /10000 (转为百米单位)，如果你要标准米，请除以 100
    
    depth_in_meters = image / 10000.0  # cm -> meters
    
    return depth_in_meters