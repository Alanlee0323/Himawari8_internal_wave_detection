import json
import numpy as np
from PIL import Image, ImageDraw
import os
import glob

def create_mask(json_file, img_size, output_file):
    # 如果JSON文件存在，創建對應的mask
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        mask = Image.new('L', img_size, 0)
        draw = ImageDraw.Draw(mask)
        
        for shape in data['shapes']:
            points = shape['points']
            points = [tuple(point) for point in points]
            
            if shape['shape_type'] == 'polygon':
                draw.polygon(points, fill=255)
            elif shape['shape_type'] == 'rectangle':
                draw.rectangle(points, fill=255)
    else:
        # 如果JSON文件不存在，創建一個空的mask
        mask = Image.new('L', img_size, 0)
    
    mask.save(output_file)
    print(f"Mask saved to {output_file}")

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 獲取所有圖片文件
    image_files = glob.glob(os.path.join(input_dir, '*.jpg')) + glob.glob(os.path.join(input_dir, '*.png'))
    
    for image_file in image_files:
        # 獲取圖片大小
        with Image.open(image_file) as img:
            img_size = img.size
        
        # 生成對應的JSON文件名和輸出mask文件名
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        json_file = os.path.join(input_dir, f"{base_name}.json")
        output_file = os.path.join(output_dir, f"{base_name}_mask.png")
        
        # 創建mask（無論是否存在對應的JSON文件）
        create_mask(json_file, img_size, output_file)

# 使用示例
input_dir = r'C:\Users\Alan\Dropbox\Himawari8_Wave_Speed_Detection\datas\Band03IW\Original_imgs\NLM\Labeled\20210624_20210630'
output_dir = r'C:\Users\Alan\Dropbox\Himawari8_Wave_Speed_Detection\datas\Band03IW\Original_imgs\NLM\Labeled\20210624_20210630'

process_directory(input_dir, output_dir)