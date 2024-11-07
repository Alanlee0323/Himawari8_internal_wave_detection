import numpy as np
import cv2
import matplotlib.pyplot as plt

# 創建卷積核
def create_wave_kernel(size=21):
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    kernel[size//2, :] = -1
    return kernel

# 應用內波增強與雲層抑制
def apply_wave_convolution(image, window_size=9, reduction_factor=0.5):
    h, w = image.shape
    output_image = image.copy()
    avg_brightness_list = []

    # 逐塊掃描圖像，計算每個窗口的平均亮度
    for i in range(0, h - window_size + 1, window_size):
        for j in range(0, w - window_size + 1, window_size):
            # 提取局部區域
            local_region = image[i:i + window_size, j:j + window_size]
            
            # 計算該區域的平均亮度
            avg_brightness = np.mean(local_region)
            
            # 將平均亮度存儲起來以計算全局閾值
            avg_brightness_list.append(avg_brightness)
    
    # 計算全局平均亮度作為自動閾值
    global_threshold = np.mean(avg_brightness_list)*1.5
    print(f"Calculated Global Threshold: {global_threshold}")

    # 再次掃描圖像，根據全局閾值壓低亮度
    for i in range(0, h - window_size + 1, window_size):
        for j in range(0, w - window_size + 1, window_size):
            # 提取局部區域
            local_region = image[i:i + window_size, j:j + window_size]
            
            # 計算該區域的平均亮度
            avg_brightness = np.mean(local_region)
            
            # 如果該區域的亮度高於全局閾值，則壓低亮度
            if avg_brightness > global_threshold:
                output_image[i:i + window_size, j:j + window_size] = (
                    local_region * reduction_factor
                ).astype(image.dtype)
    
    return np.clip(output_image, 0, 255)

# 畫直方圖的輔助函數
def plot_histogram(ax, image, title):
    ax.hist(image.ravel(), bins=256, range=(0, 256), density=True, alpha=0.7)
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    ax.set_title(title)

# 載入數據
band = np.loadtxt(r'C:\Users\Alan\Dropbox\Himawari8_Wave_Speed_Detection\Try_band\tbb-201906180450.ext.01.fld_output.txt')

# 確保數據在0-255範圍內
band_scaled = np.interp(band, (band.min(), band.max()), (0, 255)).astype(np.uint8)
print(f"Original band scaled min: {band_scaled.min()}, max: {band_scaled.max()}")

# 應用內波增強和雲層抑制
enhanced_image = apply_wave_convolution(band_scaled, window_size=5, reduction_factor=0.5)

# 創建一個2x3的子圖佈局
fig, axs = plt.subplots(2, 3, figsize=(20, 15))

# 顯示原始圖像及其直方圖
axs[0, 0].imshow(band_scaled, cmap='gray')
axs[0, 0].set_title('Original Band')
plot_histogram(axs[0, 1], band_scaled, 'Original Band Histogram')

# 顯示增強後的圖像及其直方圖
axs[1, 0].imshow(enhanced_image, cmap='gray')
axs[1, 0].set_title('Enhanced Internal Wave Features')
plot_histogram(axs[1, 1], enhanced_image, 'Enhanced Image Histogram')

# 顯示差異圖及其直方圖
difference = enhanced_image.astype(np.float32) - band_scaled.astype(np.float32)
axs[0, 2].imshow(difference, cmap='coolwarm', vmin=-50, vmax=50)
axs[0, 2].set_title('Difference (Enhanced - Original)')
plot_histogram(axs[1, 2], difference + 128, 'Difference Histogram')

# 調整佈局並顯示
plt.tight_layout()
plt.show()

# 保存結果
cv2.imwrite('enhanced_internal_wave.png', enhanced_image)
cv2.imwrite('enhancement_difference.png', ((difference + 50) / 100 * 255).astype(np.uint8))
