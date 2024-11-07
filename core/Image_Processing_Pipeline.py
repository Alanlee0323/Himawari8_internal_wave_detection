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

# 現有的函數保持不變
def read_data(file_path):
    data = np.loadtxt(file_path)
    return data

def gamma_correction(image, gamma=1.0):
    if image.dtype != np.float32:
        image = image.astype(np.float32)
        image_min = np.min(image)
        image_max = np.max(image)
        if image_min != image_max:
            image = (image - image_min) / (image_max - image_min)
        else:
            return np.zeros_like(image, dtype=np.uint8)
    
    corrected = np.power(image, gamma)
    corrected = np.clip(corrected * 255, 0, 255)
    return corrected.astype(np.uint8)

def msr_process(image, scales=[5, 15, 30, 60]):
    def MSR(img, scales):
        result = np.zeros_like(img, dtype=np.float32)
        for scale in scales:
            blurred = cv2.GaussianBlur(img, (0, 0), scale)
            result += np.log1p(img) - np.log1p(blurred)
        result = result / len(scales)
        return result

    msr_image = MSR(image, scales)
    msr_image = cv2.normalize(msr_image, None, 0, 255, cv2.NORM_MINMAX)
    return msr_image.astype(np.uint8)

def plot_with_geo_info(image, extent, ax):
    # 移除所有額外元素，只保留圖像本身
    ax.set_position([0, 0, 1, 1])
    ax.set_extent(extent)
    
    img = ax.imshow(image, extent=extent, transform=ccrs.PlateCarree(), 
                    cmap='gray', norm=PowerNorm(gamma=0.8), origin='upper')
    
    # 移除所有軸線和標籤
    ax.axis('off')
    
    # 移除邊框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

# def plot_with_geo_info(image, extent, time_info, locked_longitude, locked_latitude, title, ax):
#     # 設置緊湊布局為 False，防止自動調整大小
#     plt.tight_layout(False)
    
#     # 設置子圖的位置，使用完整空間
#     ax.set_position([0.1, 0.1, 0.8, 0.8])
#     ax.set_extent(extent)
#     ax.add_feature(cfeature.COASTLINE)
#     ax.add_feature(cfeature.BORDERS, linestyle=':')
    
#     img = ax.imshow(image, extent=extent, transform=ccrs.PlateCarree(), 
#                     cmap='gray', norm=PowerNorm(gamma=0.8), origin='upper')
    
#     cbar = plt.colorbar(img, ax=ax, orientation='vertical', pad=0.03, aspect=15, shrink=0.3)
#     cbar.set_label('Value')
    
#     # gl = ax.gridlines(draw_labels=True, linewidth=1, color='white', alpha=0.5, linestyle='--')
#     # gl.top_labels = False
#     # gl.right_labels = False
#     # gl.xformatter = LONGITUDE_FORMATTER
#     # gl.yformatter = LATITUDE_FORMATTER
    
#     ax.set_xticks([])
#     ax.set_yticks([])

#     ax.plot(locked_longitude, locked_latitude, marker='o', color='red', markersize=4, transform=ccrs.PlateCarree())
    
#     ax.text(0.95, 0.95, time_info + ' UTC', horizontalalignment='right', 
#              verticalalignment='top', transform=ax.transAxes, fontsize=10, 
#              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
#     ax.set_title(title)

# 新的處理函數
def process_and_save_image(file_path, output_folder, process_type):
    print(f"Processing file: {file_path}")
    raw_data = read_data(file_path)
    
    # NLM (使用 gamma 校正模擬 Non-Local Means)
    nlm_image = gamma_correction(raw_data, gamma=0.8)
    
    if process_type == 'NLM':
        processed_image = nlm_image
    elif process_type == 'MSR':
        processed_image = msr_process(nlm_image)
    elif process_type == 'MSRCLAHE':
        msr_image = msr_process(nlm_image)
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2))
        processed_image = clahe.apply(msr_image)
    else:
        raise ValueError(f"Unknown process type: {process_type}")

    # 繪圖和保存
    extent = [119, 120, 20, 21]
    time_info = re.search(r'\d{4}\d{2}\d{2}\d{2}\d{2}', file_path).group()
    locked_longitude, locked_latitude = 119.25165, 20.60495
    
    # 創建無邊框的圖形
    fig = plt.figure(figsize=(15, 8), frameon=False)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # plot_with_geo_info(processed_image, extent, time_info, locked_longitude, locked_latitude, f'{process_type} Processed Image', ax)
    plot_with_geo_info(processed_image, extent, ax)
    # 保存時禁用自動調整
    plt.savefig(os.path.join(output_folder, f"{time_info}.png"), 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0,
                transparent=True)
    plt.close(fig)

# 主處理流程
def main_processing_pipeline(original_folder, output_base_folder):
    process_types = ['NLM', 'MSR', 'MSRCLAHE']
    
    for root, dirs, files in os.walk(original_folder):
        for dir_name in dirs:
            input_subfolder = os.path.join(root, dir_name)
            
            for process_type in process_types:
                output_subfolder = os.path.join(output_base_folder, process_type, dir_name)
                os.makedirs(output_subfolder, exist_ok=True)
                
                txt_files = glob.glob(os.path.join(input_subfolder, '*.txt'))
                for txt_file in txt_files:
                    process_and_save_image(txt_file, output_subfolder, process_type)
    
    print("All images have been processed and saved.")

# 執行主處理流程
original_folder = r'C:\Users\Alan\Dropbox\Himawari8_Wave_Speed_Detection\datas\Band03IW\Label_txt_1'
output_base_folder = r'C:\Users\Alan\Dropbox\Himawari8_Wave_Speed_Detection\datas\Band03IW\Label_txt_1'
main_processing_pipeline(original_folder, output_base_folder)

# 可選：為每個處理類型創建視頻
def create_video(folder_path, output_name, fps=6):
    png_files = sorted(glob.glob(os.path.join(folder_path, '*.png')))
    if png_files:
        frame = cv2.imread(png_files[0])
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(folder_path, f'{output_name}.mp4')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        for png_file in png_files:
            frame = cv2.imread(png_file)
            out.write(frame)

        out.release()
        print(f"Video has been created: {video_path}")
    else:
        print(f"No PNG files found in {folder_path}")

# 為每種處理類型創建視頻
for process_type in ['NLM', 'MSR', 'MSRCLAHE']:
    for root, dirs, files in os.walk(os.path.join(output_base_folder, process_type)):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            create_video(folder_path, f'{process_type}_{dir_name}_video')