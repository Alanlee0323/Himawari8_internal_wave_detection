import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

def create_augmentation_pipeline():
    """創建擴增管線"""
    transform = A.Compose([
        # 亮度和對比度調整
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.7
        ),
        
        # 加入噪聲
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
        ], p=0.4),
        
        # 局部變形（輕微）
        A.OneOf([
            A.GridDistortion(distort_limit=0.1, p=0.5),
            A.ElasticTransform(alpha=1, sigma=10, alpha_affine=10, p=0.5)
        ], p=0.3),
        
        # 小區域遮罩
        A.CoarseDropout(
            max_holes=8,
            max_height=8,
            max_width=8,
            min_holes=5,
            min_height=5,
            min_width=5,
            p=0.3
        ),
    ])
    return transform

def augment_dataset(input_dir, output_dir, num_augmentations=3):
    """
    對數據集進行擴增
    
    Args:
        input_dir: 輸入資料夾路徑
        output_dir: 輸出資料夾路徑
        num_augmentations: 每張圖片擴增的數量
    """
    # 確保輸出資料夾存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 創建擴增管線
    transform = create_augmentation_pipeline()
    
    # 獲取所有圖片檔案
    files = os.listdir(input_dir)
    mask_files = [f for f in files if '_mask' in f]
    original_files = [f.replace('_mask', '') for f in mask_files]
    
    print(f"找到 {len(mask_files)} 對圖片進行擴增...")
    
    # 對每對圖片進行擴增
    for mask_file, orig_file in tqdm(zip(mask_files, original_files), total=len(mask_files)):
        # 讀取圖片
        mask_path = os.path.join(input_dir, mask_file)
        orig_path = os.path.join(input_dir, orig_file)
        
        mask_img = cv2.imread(mask_path)
        orig_img = cv2.imread(orig_path)
        
        if mask_img is None or orig_img is None:
            print(f"警告：無法讀取圖片對 {mask_file} 和 {orig_file}")
            continue
        
        # 進行多次擴增
        for i in range(num_augmentations):
            # 應用相同的擴增到兩張圖片
            transformed = transform(image=orig_img, mask=mask_img)
            aug_orig = transformed['image']
            aug_mask = transformed['mask']
            
            # 生成新的檔名
            base_name = os.path.splitext(orig_file)[0]
            aug_orig_name = f"{base_name}_aug_{i+1}.png"
            aug_mask_name = f"{base_name}_aug_{i+1}_mask.png"
            
            # 儲存擴增後的圖片
            cv2.imwrite(os.path.join(output_dir, aug_orig_name), aug_orig)
            cv2.imwrite(os.path.join(output_dir, aug_mask_name), aug_mask)
    
    print("擴增完成！")

if __name__ == "__main__":
    input_directory = r"C:\Users\Alan\Dropbox\Himawari8_Wave_Speed_Detection\datas\Band03IW\Original_imgs\NLM\Labeled\20241024_Train_IW"
    output_directory = os.path.join(input_directory, "augmented")
    
    try:
        augment_dataset(
            input_dir=input_directory,
            output_dir=output_directory,
            num_augmentations=3  # 每張圖片擴增3次
        )
        
        # 統計結果
        original_files = len([f for f in os.listdir(input_directory) if '_mask' in f])
        augmented_files = len([f for f in os.listdir(output_directory) if '_mask' in f])
        
        print(f"\n擴增統計：")
        print(f"原始圖片對數量: {original_files}")
        print(f"擴增後圖片對數量: {augmented_files}")
        print(f"總圖片對數量: {original_files + augmented_files}")
        
    except Exception as e:
        print(f"發生錯誤：{str(e)}")