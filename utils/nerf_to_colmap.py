import json
import numpy as np
import os
import shutil

def clean_filename(filepath):
    """
    清理文件路径，生成一个唯一的、扁平化的文件名。
    例如：../../train/block_1/0001.png -> train_block_1_0001.png
    """
    parts = os.path.normpath(filepath).split(os.sep)
    # 过滤掉 ".." 和 "train" (如果 "train" 总是根目录)
    cleaned_parts = [p for p in parts if p not in ('..', 'train') and p != '']
    # 将父文件夹和文件名连接起来（例如 block_1_0001.png）
    return "_".join(cleaned_parts)




def convert_nerf_to_colmap(data, sparse_path):
    w = int(data["w"])
    h = int(data["h"])
    fx = data["fl_x"]
    fy = data["fl_y"]
    cx = data.get("cx", w/2)
    cy = data.get("cy", h/2)

    # ==== 2. Write cameras.txt ====
    cameras_filepath = os.path.join(sparse_path, 'cameras.txt')
    with open(cameras_filepath, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        # Use PINHOLE (fx, fy, cx, cy)
        f.write(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")

    # COLMAP coordinate correction
    R_align = np.array([
        [1, 0, 0],
        [0,-1, 0],
        [0, 0,-1]
    ])

    # ==== 3. Write images.txt ====
    # 记录文件名对应关系
    filename_mapping = {} 
    images_filepath = os.path.join(sparse_path, 'images.txt')
    with open(images_filepath, "w") as f:
        f.write("# Image list with two lines per image:\n")
        f.write("# IMAGE_ID, QW QX QY QZ, TX TY TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[]\n")

        img_id = 1

        for frame in data["frames"]:
            original_filepath = frame["file_path"]
            new_filename = clean_filename(original_filepath)
            filename_mapping[new_filename] = original_filepath.replace("../../train/", "")

            T = np.array(frame["transform_matrix"], dtype=np.float64)  # c2w

            # === Convert c2w → w2c ===
            R = T[:3,:3]
            C = T[:3, 3]
            R_w2c = R.T
            t_w2c = -R.T @ C

            # === Convert to COLMAP camera coordinates ===
            R_colmap = R_align @ R_w2c
            t_colmap = R_align @ t_w2c

            # Convert rotation to quaternion (qw,qx,qy,qz)
            def rot_to_quat(R):
                q = np.empty(4)
                trace = np.trace(R)
                if trace > 0:
                    s = np.sqrt(trace + 1.0) * 2
                    q[0] = 0.25 * s
                    q[1] = (R[2,1] - R[1,2]) / s
                    q[2] = (R[0,2] - R[2,0]) / s
                    q[3] = (R[1,0] - R[0,1]) / s
                else:
                    i = np.argmax(np.diag(R))
                    if i == 0:
                        s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
                        q[0] = (R[2,1] - R[1,2]) / s
                        q[1] = 0.25 * s
                        q[2] = (R[0,1] + R[1,0]) / s
                        q[3] = (R[0,2] + R[2,0]) / s
                    elif i == 1:
                        s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
                        q[0] = (R[0,2] - R[2,0]) / s
                        q[1] = (R[0,1] + R[1,0]) / s
                        q[2] = 0.25 * s
                        q[3] = (R[1,2] + R[2,1]) / s
                    else:
                        s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
                        q[0] = (R[1,0] - R[0,1]) / s
                        q[1] = (R[0,2] + R[2,0]) / s
                        q[2] = (R[1,2] + R[2,1]) / s
                        q[3] = 0.25 * s
                return q / np.linalg.norm(q)

            q = rot_to_quat(R_colmap)

            f.write(f"{img_id} {q[0]} {q[1]} {q[2]} {q[3]} "
                    f"{t_colmap[0]} {t_colmap[1]} {t_colmap[2]} "
                    f"1 {new_filename}\n")
            f.write("\n")   # no 2D-3D matches
            
            if img_id == 1:
                print("--- COLMAP W2C Poses ---")
                print(f"Rotation Matrix R_W2C (COLMAP):\n{R_colmap.round(4)}")
                print(f"Translation T_W2C (COLMAP): {t_colmap.round(4)}")
                print(f"Quaternion [QW, QX, QY, QZ]: {np.array([q[0], q[1], q[2], q[3]]).round(10)}")
            img_id += 1

    # ==== 4. Empty points3D.txt ====
    images_filepath = os.path.join(sparse_path, "points3D.txt")
    with open(images_filepath, "w") as f:
        f.write("# Empty 3d points\n")

    return filename_mapping



image_original_path = "/media/rylynn/data/MatrixCity/"
current_path = "/home/rylynn/Pictures/datasets_3Dline/MatrixCity/block_B"
image_path = os.path.join(current_path, 'images')
os.makedirs(image_path, exist_ok=True)
sparse_path = os.path.join(current_path, 'sparse_original')
os.makedirs(sparse_path, exist_ok=True)
json_path = os.path.join(current_path, 'transforms_train.json')


# ==== 1. Load JSON ====
with open(json_path, "r") as f:
    data = json.load(f)

# 执行转换
filename_mapping = convert_nerf_to_colmap(data, sparse_path)
# 复制图片
for new_filename, original_filepath in filename_mapping.items():
    # 构造原始文件的完整路径
    original_file_fullpath = os.path.join(image_original_path, original_filepath)
    
    # 构造目标文件的完整路径
    target_file_fullpath = os.path.join(image_path, new_filename)
    
    # 确保目标目录存在
    os.makedirs(os.path.dirname(target_file_fullpath), exist_ok=True)
    
    # 复制文件
    shutil.copy2(original_file_fullpath, target_file_fullpath)

print(f"✅ Images from {original_file_fullpath} copied to: {image_path}")