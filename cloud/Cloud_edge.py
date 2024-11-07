import os
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import PowerNorm
import glob


def read_data(file_path):
    return np.loadtxt(file_path)

def enhance_image(image):
    # 將圖像轉換為 uint8 類型
    image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    # 應用 CLAHE（對比度受限自適應直方圖均衡化）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image_uint8)
    return enhanced

def remove_clouds(image, k=0.8, inpaint_radius=3):
    # 轉換圖像到 uint8 類型
    image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    
    # 計算閾值並創建掩碼
    threshold = np.mean(image_uint8) + k * np.std(image_uint8)
    mask = (image_uint8 > threshold).astype(np.uint8) * 255
    
    # 使用 inpaint 填充被遮罩的區域
    cloud_removed = cv2.inpaint(image_uint8, mask, inpaint_radius, cv2.INPAINT_NS)
    
    return cloud_removed, mask


def detect_edges(image, low_threshold=300, high_threshold=500):
    # 確保圖像是 uint8 類型
    image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    edges = cv2.Canny(image_uint8, low_threshold, high_threshold, L2gradient=True)
    return edges

def plot_with_geo_info(image, extent, time_info, locked_longitude, locked_latitude, title, ax, vmin=None, vmax=None):
    ax.set_extent(extent)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # 如果提供了 vmin 和 vmax，則在 PowerNorm 中使用它們
    if vmin is not None and vmax is not None:
        norm = PowerNorm(gamma=0.8, vmin=vmin, vmax=vmax)
    else:
        norm = PowerNorm(gamma=0.8)
    
    img = ax.imshow(image, extent=extent, transform=ccrs.PlateCarree(), 
                    cmap='gray', norm=norm, origin='upper')
    cbar = plt.colorbar(img, ax=ax, orientation='vertical', pad=0.03, aspect=15, shrink=0.3)
    cbar.set_label('Value')
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='white', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    ax.plot(locked_longitude, locked_latitude, marker='o', color='red', markersize=4, transform=ccrs.PlateCarree())
    ax.text(0.95, 0.95, time_info + ' UTC', horizontalalignment='right', 
            verticalalignment='top', transform=ax.transAxes, fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax.set_title(title)

def filter_edges_by_angle(edges, min_angle=85, max_angle=110, min_line_length=50, max_line_gap=10):
    # 霍夫線變換
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    if lines is None:
        return np.zeros_like(edges)

    # 創建一個空白mask
    mask = np.zeros_like(edges)
    
    # 根據角度過濾線段
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if min_angle <= angle <= max_angle or min_angle <= (180 - angle) <= max_angle:
            filtered_lines.append(line[0])
    
    # 對過濾後的線段進行排序和合併
    filtered_lines.sort(key=lambda line: (line[0], line[1]))  # 按起點排序
    merged_lines = []
    for line in filtered_lines:
        if not merged_lines:
            merged_lines.append(line)
        else:
            last_line = merged_lines[-1]
            if np.sqrt((line[0] - last_line[2])**2 + (line[1] - last_line[3])**2) < max_line_gap:
                # 如果兩線段夠近，合併它們
                merged_lines[-1] = (last_line[0], last_line[1], line[2], line[3])
            else:
                merged_lines.append(line)
    
    # 繪製合併後的線段
    for line in merged_lines:
        x1, y1, x2, y2 = line
        cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
    
    # 應用形態學操作來平滑線段
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    # 將掩碼應用到原始邊緣圖像
    filtered_edges = cv2.bitwise_and(edges, mask)
    
    return filtered_edges

# 在 process_image 函數中更新調用
def process_image(image_path):
    raw_data = read_data(image_path)
    
    # 增強圖像
    enhanced = enhance_image(raw_data)
    
    # 移除雲層並填充
    cloud_removed, cloud_mask = remove_clouds(enhanced, k=0.8, inpaint_radius=3)
    
    # 邊緣檢測
    edges = detect_edges(cloud_removed)
    filtered_edges = filter_edges_by_angle(edges)
    
    # 將邊緣疊加到雲層處理後的圖像上
    edge_overlay = np.dstack([cloud_removed]*3)  # 創建三通道圖像，保持原始值
    edge_overlay[filtered_edges != 0] = [0, 255, 255]  # 黃色邊緣
    
    return raw_data, enhanced, cloud_removed, edge_overlay

def plot_and_save(raw_data, enhanced, cloud_removed, edge_overlay, time_info, output_path):
    extent = [119, 120, 20, 21]
    locked_longitude, locked_latitude = 119.25165, 20.60495
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 20), subplot_kw={'projection': ccrs.PlateCarree()})
    
    vmin, vmax = np.min(cloud_removed), np.max(cloud_removed)  # 使用相同的值範圍
    
    # plot_with_geo_info(raw_data, extent, time_info, locked_longitude, locked_latitude, 'Original Image', axs[0, 0])
    # plot_with_geo_info(enhanced, extent, time_info, locked_longitude, locked_latitude, 'Enhanced Image', axs[0, 1])
    # plot_with_geo_info(cloud_removed, extent, time_info, locked_longitude, locked_latitude, 'Cloud Removed', axs[1, 0], vmin=vmin, vmax=vmax)
    plot_with_geo_info(edge_overlay[:,:,0], extent, time_info, locked_longitude, locked_latitude, 'Edge Detection', axs, vmin=vmin, vmax=vmax)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# 主處理流程
input_dir = r'C:\Users\Alan\Dropbox\Himawari8_Wave_Speed_Detection\datas\Band03txt\20190617_20190618'
output_dir = r'C:\Users\Alan\Dropbox\Himawari8_Wave_Speed_Detection\output5'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        input_path = os.path.join(input_dir, filename)
        time_info = re.search(r'\d{4}\d{2}\d{2}\d{2}\d{2}', filename).group()
        output_path = os.path.join(output_dir, f"{time_info}.png")
        
        print(f"Processing file: {filename}")
        raw_data, enhanced, cloud_removed, edge_overlay = process_image(input_path)
        plot_and_save(raw_data, enhanced, cloud_removed, edge_overlay, time_info, output_path)
        print(f"Processed image saved to: {output_path}")

print("All files have been processed.")

# 將PNG圖片組合成MP4影片
png_files = sorted(glob.glob(os.path.join(output_dir, '*.png')))
if png_files:
    frame = cv2.imread(png_files[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(output_dir, 'output_video.mp4')
    out = cv2.VideoWriter(video_path, fourcc, 6, (width, height))
    for png_file in png_files:
        frame = cv2.imread(png_file)
        out.write(frame)
    out.release()
    print(f"Video has been created: {video_path}")
else:
    print("No PNG files found to create video.")