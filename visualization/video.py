import cv2
import os
from natsort import natsorted

def create_video_from_images(input_folder, output_folder, output_filename, fps):
    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 獲取輸入資料夾中的所有圖片文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files = natsorted(image_files)  # 自然排序文件名
    
    if not image_files:
        print("沒有找到圖片文件。")
        return
    
    # 讀取第一張圖片來獲取尺寸
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, layers = first_image.shape
    
    # 設定影片編碼器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(output_folder, output_filename), fourcc, fps, (width, height))
    
    # 讀取每張圖片並寫入影片
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        video.write(frame)
    
    # 釋放資源
    video.release()
    print(f"影片已保存到 {os.path.join(output_folder, output_filename)}")

if __name__ == "__main__":
    # 在這裡直接設定參數
    input_folder = r"C:\Users\Alan\Downloads\comparison_results"  # 輸入資料夾路徑
    output_folder = r"C:\Users\Alan\Downloads\comparison_results"  # 輸出資料夾路徑
    output_filename = "output.mp4"  # 輸出影片檔名
    fps = 2  # 影片的每秒幀數
    
    create_video_from_images(input_folder, output_folder, output_filename, fps)