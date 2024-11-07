import os
import shutil

def move_txt_files(source_folder, destination_folder):
    """
    # 移動指定資料夾中的所有txt檔案到指定資料夾
    # 這個程式會遍歷指定資料夾中的所有資料夾和子資料夾，並將所有txt檔案移動到指定資料夾中
    # source_folder: 要遍歷的資料夾路徑
    # destination_folder: 要將txt檔案移動到的目標資料夾路徑
    """

    # 檢查目標資料夾是否存在，如果不存在則建立
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 遍歷指定資料夾中的所有資料夾和子資料夾
    for root, _, files in os.walk(source_folder):
        # 遍歷每個資料夾中的檔案
        for file in files:
            # 只處理txt檔案
            if file.endswith(".txt"):
                # 構建檔案的完整路徑
                file_path = os.path.join(root, file)
                # 移動檔案到目標資料夾
                shutil.move(file_path, destination_folder)
                print(f"Moved {file_path} to {destination_folder}")

# 指定要處理的資料夾路徑
source_folder = r"C:\Users\Alan\Dropbox\Himawari8_Wave_Speed_Detection\datas\Band03IW\new_轉換\20210626"
destination_folder = r"C:\Users\Alan\Dropbox\Himawari8_Wave_Speed_Detection\datas\Band03IW\new_轉換\20210626"

# 執行移動檔案的函式
move_txt_files(source_folder, destination_folder)