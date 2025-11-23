import numpy as np
import cv2
from pytlsd import lsd
from afm_op import afm
import torch
from skimage.draw import line

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
