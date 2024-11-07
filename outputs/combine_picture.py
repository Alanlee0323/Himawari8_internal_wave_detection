import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 設定資料夾路徑和每張圖片的行數和列數
folder_path = r'C:\Users\Alan\Dropbox\Himawari8_Wave_Speed_Detection\image_processing\Multibands'  # 請替換成你的資料夾路徑
rows_per_image = 7  # 每張圖像的行數
cols_per_image = 2  # 每張圖像的列數

# 讀取資料夾內所有圖片
images = []
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith('.png'):
        img = Image.open(os.path.join(folder_path, filename))
        images.append(img)

# 總圖片數量
total_images = len(images)
images_per_combined_image = rows_per_image * cols_per_image
num_combined_images = (total_images + images_per_combined_image - 1) // images_per_combined_image

# 生成並保存每張組合圖像
for i in range(num_combined_images):
    fig, axs = plt.subplots(rows_per_image, cols_per_image, figsize=(15, 15))
    axs = axs.flatten()

    # 將圖片填充到子圖中
    for j in range(rows_per_image * cols_per_image):
        idx = i * images_per_combined_image + j
        if idx < total_images:
            axs[j].imshow(np.array(images[idx]))
            axs[j].axis('off')
        else:
            axs[j].axis('off')

    plt.tight_layout()
    plt.savefig(f'combined_image_{i + 1}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

print(f"共生成 {num_combined_images} 張圖像")
