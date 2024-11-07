import os
import glob

def delete_augmented_images(folder_path):
    """
    刪除指定資料夾中所有含有 '_aug' 關鍵字的影像檔案
    
    Parameters:
    folder_path (str): 要處理的資料夾路徑
    
    Returns:
    tuple: (刪除的檔案數量, 刪除的檔案列表)
    """
    # 確認資料夾是否存在
    if not os.path.exists(folder_path):
        raise ValueError(f"資料夾路徑不存在: {folder_path}")
    
    # 取得所有含有 '_aug' 的檔案路徑
    # 支援常見的影像副檔名
    aug_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
        pattern = os.path.join(folder_path, f'*_aug*{ext}')
        aug_files.extend(glob.glob(pattern))
        # 不區分大小寫的檔案副檔名
        pattern = os.path.join(folder_path, f'*_aug*{ext.upper()}')
        aug_files.extend(glob.glob(pattern))
    
    # 保存要刪除的檔案清單
    deleted_files = []
    
    # 刪除檔案
    for file_path in aug_files:
        try:
            os.remove(file_path)
            deleted_files.append(os.path.basename(file_path))
        except Exception as e:
            print(f"刪除檔案時發生錯誤 {file_path}: {str(e)}")
    
    return len(deleted_files), deleted_files

# 使用範例
if __name__ == "__main__":
    # 替換為你的資料夾路徑
    folder_path = r"C:\Users\Alan\Dropbox\Himawari8_Wave_Speed_Detection\datas\Band03IW\Original_imgs\NLM\Labeled\20241024_Train_IW\mask"
    
    try:
        count, deleted = delete_augmented_images(folder_path)
        print(f"成功刪除 {count} 個擴增影像檔案")
        print("\n刪除的檔案列表:")
        for file in deleted:
            print(f"- {file}")
            
    except Exception as e:
        print(f"執行過程中發生錯誤: {str(e)}")