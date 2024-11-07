import os

def delete_png_mp4_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.mp4')):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"已刪除: {file_path}")
                except Exception as e:
                    print(f"無法刪除 {file_path}: {e}")

if __name__ == "__main__":
    # 在這裡指定目標資料夾的路徑
    target_directory = r"C:\Users\Alan\Dropbox\Himawari8_Wave_Speed_Detection\datas\Band03IW\Original"
    
    if not os.path.isdir(target_directory):
        print(f"錯誤: {target_directory} 不是一個有效的資料夾路徑")
    else:
        print(f"正在處理資料夾: {target_directory}")
        delete_png_mp4_files(target_directory)
        print("處理完成")