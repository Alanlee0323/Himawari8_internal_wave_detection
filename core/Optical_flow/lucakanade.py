import cv2
import numpy as np
import matplotlib.pyplot as plt

def build_pyramid(image, levels):
    """建立圖像金字塔"""
    pyramid = [image]
    for _ in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def select_features(image, max_corners=1000, quality_level=0.01, min_distance=10):
    """選擇特徵點"""
    features = cv2.goodFeaturesToTrack(image, max_corners, quality_level, min_distance)
    if features is not None:
        print(f"選擇到的特徵點數量: {len(features)}")
        return features.reshape(-1, 2)
    else:
        print("未選擇到任何特徵點")
        return np.array([]).reshape(-1, 2)

def lucas_kanade_optical_flow(img1, img2, points, window_size=15, max_level=3):
    """計算 Lucas-Kanade 光流"""
    lk_params = dict(winSize=(window_size, window_size),
                     maxLevel=max_level,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    new_points, status, _ = cv2.calcOpticalFlowPyrLK(img1, img2, points, None, **lk_params)
    print(f"計算到的新特徵點數量: {len(new_points)}")
    print(f"狀態: {status.ravel()}")
    return new_points, status

def visualize_flow(image, points, new_points, status):
    """視覺化光流結果"""
    mask = np.zeros_like(image)
    for i, (new, old) in enumerate(zip(new_points, points)):
        if status[i]:
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
    return cv2.add(image, mask)

def detect_internal_waves_from_video(video_path, pyramid_levels=3):
    """從影片中檢測內波"""
    cap = cv2.VideoCapture(video_path)
    
    # 檢查是否成功打開影片
    if not cap.isOpened():
        print("無法打開影片")
        return
    
    ret, first_frame = cap.read()
    if not ret:
        print("無法讀取影片的第一幀")
        return
    
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_pyramid = build_pyramid(prev_gray, pyramid_levels)
    
    features = select_features(prev_gray)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("影片結束或無法讀取幀")
            break
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_pyramid = build_pyramid(curr_gray, pyramid_levels)
        
        # 金字塔 LK 光流計算
        new_features = features.copy()
        status = np.ones(len(features), dtype=bool)
        for level in reversed(range(pyramid_levels)):
            if level < pyramid_levels - 1:
                new_features *= 2
                new_features = new_features[status]
            
            new_features, sub_status = lucas_kanade_optical_flow(
                prev_pyramid[level], curr_pyramid[level], new_features
            )
            status = status & sub_status.ravel()
        
        # 過濾有效特徵點
        good_new = new_features[status]
        good_old = features[status]
        print(f"有效特徵點數量: {len(good_new)}")
        
        # 視覺化結果
        vis_image = visualize_flow(frame, good_old, good_new, status)
        
        # 顯示結果
        cv2.imshow('Optical Flow', vis_image)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        
        # 更新特徵點
        features = good_new.reshape(-1, 1, 2)
        prev_gray = curr_gray
        prev_pyramid = curr_pyramid
    
    cap.release()
    cv2.destroyAllWindows()

# 使用示例
def main():
    video_path = r'C:\Users\Alan\Dropbox\Himawari8_Wave_Speed_Detection\datas\Band03IW\20190617_20190618\.mp4'
    
    detect_internal_waves_from_video(video_path)

if __name__ == '__main__':
    main()
