import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def process_image(image_path):
    # 讀取圖像數據
    image = np.loadtxt(image_path)
    
    # 計算閾值
    threshold = np.mean(image) * 1.3
    
    # 二值化圖像處理
    binary_image = np.where(image >= threshold, 255, 0).astype(np.uint8)
    
    # 相關性處理
    processed_image = np.where(binary_image == 255, image.min(), image)
    
    return image, binary_image, processed_image, threshold

def plot_and_save(image, binary_image, processed_image, threshold, output_path):
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    
    # 顯示原始圖像
    im0 = ax[0, 0].imshow(image, cmap='gray', vmin=image.min(), vmax=image.max())
    ax[0, 0].set_title('Original')
    fig.colorbar(im0, ax=ax[0, 0])
    
    # 顯示二值化圖像
    ax[0, 1].imshow(binary_image, cmap='gray')
    ax[0, 1].set_title('Binary')
    
    # 顯示處理後的圖像
    im2 = ax[1, 0].imshow(processed_image, cmap='gray', vmin=image.min(), vmax=image.max())
    ax[1, 0].set_title('Processed')
    fig.colorbar(im2, ax=ax[1, 0])
    
    # 顯示直方圖
    ax[1, 1].hist(image.ravel(), bins=256, range=(image.min(), image.max()), color='gray', alpha=0.5, label='原始圖像')
    ax[1, 1].hist(processed_image.ravel(), bins=256, range=(image.min(), image.max()), color='blue', alpha=0.5, label='處理後圖像')
    ax[1, 1].set_title('Histogram')
    ax[1, 1].set_xlabel('Pixel')
    ax[1, 1].set_ylabel('Freq')
    ax[1, 1].axvline(x=threshold, color='red', linestyle='--', label='Threshhold')
    ax[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# 設置輸入和輸出目錄
input_dir = r'C:\Users\Alan\Dropbox\Himawari8_Wave_Speed_Detection\datas\Band03txt\20190617_20190618'
output_dir = r'C:\Users\Alan\Dropbox\Himawari8_Wave_Speed_Detection\output'

# 如果輸出目錄不存在，則創建它
os.makedirs(output_dir, exist_ok=True)

# 處理輸入目錄中的所有 txt 文件
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_processed.png")
        
        print(f"正在處理文件：{filename}")
        image, binary_image, processed_image, threshold = process_image(input_path)
        plot_and_save(image, binary_image, processed_image, threshold, output_path)
        print(f"處理後的圖像已保存至：{output_path}")

print("所有文件已處理完成。")