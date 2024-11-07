import cv2
import numpy as np
import os

def dense_optical_flow(prev_img, curr_img):
    # 將圖像轉換為灰度
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    
    # 計算密集光流
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, 
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # 將光流轉換為極坐標
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # 創建一個 HSV 圖像來表示光流
    hsv = np.zeros_like(prev_img)
    hsv[..., 1] = 255
    
    # 使用角度來表示方向，0度（由左至右）為紅色
    hsv[..., 0] = ang * 180 / np.pi
    
    # 設置閾值，只顯示接近水平方向的運動
    horizontal_threshold = 20  # 度
    mask = (np.abs(hsv[..., 0] - 180) < horizontal_threshold) | (np.abs(hsv[..., 0] - 180) < horizontal_threshold)
    
    # 調整亮度以突出水平運動
    hsv[..., 2] = np.where(mask, cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX), 0)
    
    # 將 HSV 轉換回 BGR 以便顯示
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return rgb

# 讀取影片
video_path = r'C:\Users\Alan\Dropbox\Himawari8_Wave_Speed_Detection\datas\Band03IW\20190617_20190618\output_video.mp4'
output_folder = r'C:\Users\Alan\Downloads\optical_flow_output2'  # 輸出資料夾路徑

# 確保輸出資料夾存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)

# 獲取影片的寬度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 讀取第一幀
ret, prev_frame = cap.read()
if not ret:
    print("無法讀取影片")
    exit()

frame_count = 0

while True:
    # 讀取下一幀
    ret, curr_frame = cap.read()
    if not ret:
        break

    # 計算密集光流
    flow_img = dense_optical_flow(prev_frame, curr_frame)

    # 保存結果到文件
    output_path = os.path.join(output_folder, f'flow_{frame_count:04d}.png')
    cv2.imwrite(output_path, flow_img)

    # 更新前一幀
    prev_frame = curr_frame
    frame_count += 1

    # 打印進度
    if frame_count % 10 == 0:
        print(f"已處理 {frame_count} 幀")

# 釋放資源
cap.release()

print(f"處理完成。共處理 {frame_count} 幀，結果保存在 {output_folder}")