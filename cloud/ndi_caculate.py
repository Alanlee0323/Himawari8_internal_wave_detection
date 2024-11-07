import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, filters, feature, util


def read_band_data(file_path, shape):
    with open(file_path, 'r') as file:
        data = file.read().split()
    return np.array(data, dtype=float).reshape(shape)

def downsample(array, factor):
    """簡單的降採樣函數，通過選擇每隔 factor 個像素的方式"""
    return array[::factor, ::factor]

def calculate_ndi(band03, band04):
    return np.divide(band03 - band04, band03 + band04, out=np.zeros_like(band03), where=(band03 + band04) != 0)

def save_result(result, output_file):
    with open(output_file, 'w') as file:
        for row in result:
            file.write(' '.join(map(lambda x: f"{x:.6f}", row)) + '\n')

def enhance_internal_waves(image, row_of_interest=60):
    # 確保圖像值在 -1 到 1 之間
    image = np.clip(image, -1, 1)
    
    # 1. 局部對比度增強
    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98), out_range=(-1, 1))
    
    # 2. 方向性濾波（水平方向）
    horizontal_kernel = np.array([[2, 0, -2],
                                       [2, 0, -2],
                                       [2, 0, -2]])
    img_filtered = filters.rank.mean(util.img_as_ubyte(img_rescale), footprint=horizontal_kernel)
    img_filtered = util.img_as_float64(img_filtered)
    
    # 3. 邊緣檢測
    edges = feature.canny(img_filtered, sigma=2)
    
    # 4. 高通濾波
    img_highpass = img_filtered - filters.gaussian(img_filtered, sigma=10)
    
    # 5. 突出感興趣的行
    roi = img_highpass[row_of_interest-5:row_of_interest+5, :]
    roi = np.clip(roi, -1, 1)  # 確保值在 -1 到 1 之間
    roi_enhanced = exposure.equalize_adapthist(roi)
    
    return img_rescale, img_filtered, edges, img_highpass, roi_enhanced            
    

# 新增函數用於顯示圖像
def display_images(band03, band04, ndi):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = ax1.imshow(band03, cmap='gray')
    ax1.set_title('Band03 (Down Sampling)')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(band04, cmap='gray')
    ax2.set_title('Band04')
    plt.colorbar(im2, ax=ax2)

    im3 = ax3.imshow(ndi, cmap='gray', vmin=-1, vmax=1)
    ax3.set_title('NDI')
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()
    plt.show()

# 讀取數據
band03 = read_band_data(r'C:\Users\Alan\Dropbox\1\tbb-201906180330.ext.01.fld_output.txt', (200, 200))
band04 = read_band_data(r'C:\Users\Alan\Dropbox\1\tbb-201906180330.vis.03.fld_output.txt', (100, 100))

# 將 band03 降採樣到 100x100
band03_downsampled = downsample(band03, 2)

# 確保兩個數組具有相同的形狀
assert band03_downsampled.shape == band04.shape, "形狀不匹配"

# 計算NDI
ndi = calculate_ndi(band03_downsampled, band04)

# 在主程序中，在調用 enhance_internal_waves 之前，確保 NDI 值在 -1 到 1 之間
ndi = np.clip(ndi, -1, 1)

# 應用增強函數
img_rescale, img_filtered, edges, img_highpass, roi_enhanced = enhance_internal_waves(ndi)

# 繪製結果
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(ndi, cmap='gray')
axes[0, 0].set_title('Original NDI')
axes[0, 1].imshow(img_rescale, cmap='gray')
axes[0, 1].set_title('Contrast Enhanced')
axes[0, 2].imshow(img_filtered, cmap='gray')
axes[0, 2].set_title('Directional Filtered')
axes[1, 0].imshow(edges, cmap='gray')
axes[1, 0].set_title('Edge Detection')
axes[1, 1].imshow(img_highpass, cmap='gray')
axes[1, 1].set_title('High-pass Filtered')
axes[1, 2].imshow(roi_enhanced, cmap='gray')
axes[1, 2].set_title('ROI Enhanced')

plt.tight_layout()
plt.show()

# 顯示圖像
display_images(band03_downsampled, band04, ndi)

# 保存結果
save_result(ndi, 'ndi_result.txt')

print("處理完成，結果已保存到 ndi_result.txt")