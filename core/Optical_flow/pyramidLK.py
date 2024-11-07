import cv2
import numpy as np
import os

def pyramid_lucas_kanade(prev_img, curr_img):
    # 将图像转换为灰度
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    
    # 创建网格点
    h, w = prev_gray.shape
    y, x = np.mgrid[0:h:2, 0:w:2].reshape(2, -1).astype(np.float32)
    points = np.vstack((x, y)).T
    
    # 计算金字塔 LK 光流
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, points, None, winSize=(15,15), maxLevel=3)
    
    # 计算光流向量
    flow = next_points - points
    flow = flow.reshape(h//10, w//10, 2)
    
    # 将光流转换为极坐标
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # 创建一个 HSV 图像来表示光流
    hsv = np.zeros((h//10, w//10, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    
    # 使用角度来表示方向（0-360度）
    hsv[..., 0] = ang * 180 / np.pi / 2
    
    # 使用幅度来表示亮度
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # 将 HSV 转换回 BGR 并调整大小到原始图像尺寸
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_CUBIC)
    
    return rgb

# 读取影片
video_path = r'C:\Users\Alan\Downloads\2053100-uhd_3840_2160_30fps.mp4'
output_folder = r'C:\Users\Alan\Downloads\123'  # 输出文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)

# 获取影片的宽度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 读取第一帧
ret, prev_frame = cap.read()
if not ret:
    print("无法读取影片")
    exit()

frame_count = 0

while True:
    # 读取下一帧
    ret, curr_frame = cap.read()
    if not ret:
        break

    # 计算金字塔 LK 光流
    flow_img = pyramid_lucas_kanade(prev_frame, curr_frame)

    # 保存结果到文件
    output_path = os.path.join(output_folder, f'flow_{frame_count:04d}.png')
    cv2.imwrite(output_path, flow_img)

    # 更新前一帧
    prev_frame = curr_frame
    frame_count += 1

    # 打印进度
    if frame_count % 10 == 0:
        print(f"已处理 {frame_count} 帧")

# 释放资源
cap.release()

print(f"处理完成。共处理 {frame_count} 帧，结果保存在 {output_folder}")