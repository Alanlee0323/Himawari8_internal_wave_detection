import torch
import torch.nn.functional as F
from unet import UNet
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(n_channels=4, n_classes=1)
checkpoint = torch.load(r'checkpoints/20241021/checkpoint_epoch200.pth', map_location=device)

# 移除 '_orig_mod' 前綴
new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items() if k != 'mask_values'}

model.load_state_dict(new_state_dict, strict=False)
model = torch.compile(model)  # 如果需要，重新編譯模型
model.to(device)
model.eval()

# 預處理圖片函數
def preprocess_image(image_path, scale=0.5):
    img = Image.open(image_path)
    img = img.resize((int(img.size[0]*scale), int(img.size[1]*scale)))
    img_np = np.array(img)
    if len(img_np.shape) == 2:
        img_np = np.stack([img_np] * 4, axis=0)  # 擴展到4通道
    elif img_np.shape[2] == 3:
        img_np = np.concatenate([img_np.transpose(2, 0, 1), np.zeros((1, img_np.shape[0], img_np.shape[1]))], axis=0)
    else:
        img_np = img_np.transpose((2, 0, 1))
    img_np = img_np / 255.0
    return torch.from_numpy(img_np).float().unsqueeze(0)

# 顯示結果函數
def plot_results(img, gt_mask, pred_mask, output_path):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 5))
    
    # 原始圖像
    img_np = img.squeeze().cpu().permute(1, 2, 0).numpy()
    ax1.imshow(img_np)
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    # GT 遮罩
    ax2.imshow(gt_mask, cmap='gray')
    ax2.set_title('Ground Truth Mask')
    ax2.axis('off')
    
    # 預測遮罩
    ax3.imshow(pred_mask, cmap='gray')
    ax3.set_title('Predicted Mask')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# 文件匹配函數
def match_files(img_folder, mask_folder):
    img_files = os.listdir(img_folder)
    mask_files = os.listdir(mask_folder)
    
    matched_files = []
    
    for img_file in img_files:
        img_name = os.path.splitext(img_file)[0]
        matching_mask = next((mask for mask in mask_files if img_name in mask), None)
        
        if matching_mask:
            matched_files.append((img_file, matching_mask))
        else:
            print(f"Warning: No matching mask found for {img_file}")
    
    return matched_files

# 處理整個資料夾
def process_folder(img_folder, mask_folder, output_folder):
    matched_files = match_files(img_folder, mask_folder)
    
    for img_file, mask_file in matched_files:
        img_path = os.path.join(img_folder, img_file)
        mask_path = os.path.join(mask_folder, mask_file)
        
        # 加載和預處理圖片
        image = preprocess_image(img_path)
        image = image.to(device=device, dtype=torch.float32)
        
        # 加載 GT 遮罩
        gt_mask = np.array(Image.open(mask_path).convert('L'))
        
        # 進行預測
        with torch.no_grad():
            mask_pred = model(image)
            mask_pred = torch.sigmoid(mask_pred)
            mask_pred = mask_pred.squeeze().cpu().numpy()
        
        # 後處理預測結果
        mask_pred = (mask_pred > 0.5).astype(np.uint8)
        
        # 創建輸出路徑
        output_path = os.path.join(output_folder, f'result_{os.path.splitext(img_file)[0]}.png')
        
        # 顯示並保存結果
        plot_results(image, gt_mask, mask_pred, output_path)
        print(f'Processed: {img_file}')

# 主程序
if __name__ == '__main__':
    img_folder = Path('data/imgs/')
    mask_folder = Path('data/masks/')
    output_folder = Path('results/')
    
    # 創建輸出文件夾
    output_folder.mkdir(parents=True, exist_ok=True)
    
    process_folder(img_folder, mask_folder, output_folder)
    print("Processing complete. Results saved in 'results' folder.")