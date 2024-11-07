import os
from pathlib import Path

def compare_files(imgs_dir, masks_dir):
    # 獲取兩個目錄中的所有文件名
    img_files = set(f.stem for f in Path(imgs_dir).glob('*.png'))
    mask_files = set(f.stem.replace('_mask', '') for f in Path(masks_dir).glob('*_mask.png'))

    # 找出只在其中一個目錄中出現的文件
    only_in_imgs = img_files - mask_files
    only_in_masks = mask_files - img_files

    # 打印結果
    if only_in_imgs:
        print("只在 imgs 目錄中出現的文件:")
        for file in sorted(only_in_imgs):
            print(file)

    if only_in_masks:
        print("\n只在 masks 目錄中出現的文件:")
        for file in sorted(only_in_masks):
            print(file)

    if not only_in_imgs and not only_in_masks:
        print("所有文件都配對成功！")

# 使用示例
imgs_dir = 'data/imgs'
masks_dir = 'data/masks'
compare_files(imgs_dir, masks_dir)