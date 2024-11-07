import os
import shutil

def organize_training_data(source_dir, target_subfolder="IW"):
    # 存放有標記檔案的基本名稱（不含副檔名）
    labeled_images = []
    
    # 計數器
    json_count = 0
    mask_count = 0
    image_count = 0
    
    # 取得完整的目標資料夾路徑
    target_dir = os.path.join(source_dir, target_subfolder)
    
    # 先檢查和收集所有檔案
    all_files = os.listdir(source_dir)
    
    # 第一步：找出所有的 json 檔案（代表有被標記的圖片）
    for file in all_files:
        if file.endswith('.json'):
            base_name = file.split('.')[0]  # 例如：201905090730
            labeled_images.append(base_name)
            json_count += 1
    
    # 如果有找到 json 檔案，建立目標資料夾
    if labeled_images and not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # 第二步：對每個有 json 的時間點，尋找對應的 mask 和原始圖片
    for base_name in labeled_images:
        mask_file = f"{base_name}_mask.png"
        original_file = f"{base_name}.png"
        
        # 檢查是否同時存在 mask 檔案和原始圖片
        if mask_file in all_files and original_file in all_files:
            # 複製 mask 檔案
            mask_src = os.path.join(source_dir, mask_file)
            mask_dst = os.path.join(target_dir, mask_file)
            shutil.copy2(mask_src, mask_dst)
            mask_count += 1
            
            # 複製原始圖片
            img_src = os.path.join(source_dir, original_file)
            img_dst = os.path.join(target_dir, original_file)
            shutil.copy2(img_src, img_dst)
            image_count += 1
    
    return json_count, mask_count, image_count

# 使用範例
source_directory = r"C:\Users\Alan\Dropbox\Himawari8_Wave_Speed_Detection\datas\Band03IW\Original_imgs\NLM\Labeled\20210624_20210630"

try:
    json_files, mask_files, original_files = organize_training_data(source_directory)
    print(f"統計結果：")
    print(f"找到 {json_files} 個 JSON 檔案")
    print(f"複製了 {mask_files} 個對應的 mask 檔案")
    print(f"複製了 {original_files} 個對應的原始圖片")
    
    if mask_files != json_files or original_files != json_files:
        print("\n警告：部分檔案未完整配對！")
        print(f"應該要有 {json_files} 組完整的配對檔案")
except Exception as e:
    print(f"發生錯誤：{str(e)}")