import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import PowerNorm
import os 
import re
from datetime import datetime

def plot_map_with_data_gamma(data_file_path, extent, locked_longitude, locked_latitude, time_info, figsize=(10, 8)):
    
    """
    繪製地圖並在地圖上顯示數據
    data_file_path: 包含數據的文件的路徑
    extent: 地圖的範圍，格式為[最小經度，最大經度，最小緯度，最大緯度]
    locked_longitude: 鎖定的經度
    locked_latitude: 鎖定的緯度
    time_info: 時間信息
    figsize: 圖像的大小，默認為(10, 8)
    
    """
        
    # 讀取數據
    data = np.loadtxt(data_file_path)
    print(data.shape)

    # 將數據標準化到 0-255 範圍
    data_normalized = ((data - np.min(data)) / (np.max(data) - np.min(data)) * 255).astype(np.uint8)

    # 創建圖像
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # 設置地圖範圍
    ax.set_extent(extent)

    # 添加地理特徵
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    # 繪製數據，使用標準化後的數據
    img = ax.imshow(data_normalized, extent=extent, transform=ccrs.PlateCarree(), cmap='gray', vmin=0, vmax=255, origin='upper')

    # 添加色條
    cbar = plt.colorbar(img, ax=ax, orientation='vertical', pad=0.02, aspect=50)
    cbar.set_label('灰階值')

    # 添加網格和經緯度標記
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # 畫出鎖定經緯度的紅點
    ax.plot(119.25165, 20.60495, marker='o', color='red', markersize=2, transform=ccrs.PlateCarree())

    # 在右上角添加時間資訊
    plt.text(0.95, 0.95, time_info + ' GMT', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 顯示圖像
    plt.savefig(data_file_path.replace('.txt', '.png'))
    plt.close(fig)


def main():
    # 假設經緯度範圍
    extent = [100, 140, 15, 25]
    # 鎖定的經緯度
    locked_longitude = 119
    locked_latitude = 20

    # 呼叫函數來繪製地圖
    data_folder = r'C:\Users\Alan\Downloads\1'
    pattern = re.compile(r'tbb-(\d{12})\.ext\.01\.fld_output\.txt')

    for filename in os.listdir(data_folder):
        if filename.endswith('.txt'):
            match = pattern.match(filename)
            if match:
                time_info_jst = match.group(1)
                time_info = datetime.strptime(time_info_jst, '%Y%m%d%H%M')
                time_info = time_info.strftime('%Y-%m-%d %H:%M')
                full_path = os.path.join(data_folder, filename)
                plot_map_with_data_gamma(full_path, extent, locked_longitude, locked_latitude, time_info)

# 確認這個檔案被執行，而不是被引入
if __name__ == "__main__":
    main()
