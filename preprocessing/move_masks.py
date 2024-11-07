import os
import shutil

def move_mask_files(source_dir):
    """
    將指定資料夾中所有包含 '_mask' 的檔案移動到 'mask' 子資料夾
    
    Args:
        source_dir: 來源資料夾路徑
    """
    # 建立 mask 子資料夾
    mask_dir = os.path.join(source_dir, "mask")
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    
    # 計數器
    moved_count = 0
    
    try:
        # 獲取所有檔案
        files = os.listdir(source_dir)
        
        # 找出所有包含 '_mask' 的檔案
        mask_files = [f for f in files if '_mask' in f]
        
        print(f"找到 {len(mask_files)} 個 mask 檔案")
        
        # 移動檔案
        for mask_file in mask_files:
            source_path = os.path.join(source_dir, mask_file)
            target_path = os.path.join(mask_dir, mask_file)
            
            # 確保來源檔案存在且不是資料夾
            if os.path.isfile(source_path):
                shutil.move(source_path, target_path)
                moved_count += 1
                print(f"移動: {mask_file}")
        
        print(f"\n完成！成功移動 {moved_count} 個檔案到 mask 資料夾")
        
    except Exception as e:
        print(f"發生錯誤：{str(e)}")

# 使用範例
source_directory = r"C:\Users\Alan\Dropbox\Himawari8_Wave_Speed_Detection\datas\Band03IW\Original_imgs\NLM\Labeled\20241024_Train_images"

move_mask_files(source_directory)