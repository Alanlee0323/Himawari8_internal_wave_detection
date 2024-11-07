import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_gabor_filters(image):
    gabor_responses = []
    for theta in range(0, 180, 20):  # 0到180度，每20度一次
        kernel = cv2.getGaborKernel((21, 21), 5, theta, 10, 0.5, 0, ktype=cv2.CV_32F)
        gabor_response = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        gabor_responses.append(gabor_response)
    return gabor_responses

def visualize_gabor_responses(image, gabor_responses):
    num_responses = len(gabor_responses)
    print(f"Gabor Filiter numbers{num_responses}")
    
    rows = (num_responses + 2) // 3  # 計算需要的行數
    fig, axs = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axs = axs.ravel()
    print(f"subplot：{len(axs)}")
    
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('origin')
    
    for i, response in enumerate(gabor_responses):
        print(f"處理 Gabor 響應 {i}")
        if i+1 < len(axs):
            axs[i+1].imshow(response, cmap='gray')
            axs[i+1].set_title(f'Gabor Filiter {i*20}°')
        else:
            print(f"警告：沒有足夠的子圖來顯示 Gabor 響應 {i}")
    
    # 隱藏多餘的子圖
    for i in range(num_responses + 1, len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def process_and_analyze_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"無法讀取圖像：{image_path}")
        return
    
    if image.shape[0] < 8 or image.shape[1] < 8:
        print(f"圖像太小：{image.shape}。需要至少 8x8 像素。")
        return
    
    print(f"圖像大小：{image.shape}")
    gabor_responses = apply_gabor_filters(image)
    print(f"生成了 {len(gabor_responses)} 個 Gabor 響應")
    visualize_gabor_responses(image, gabor_responses)

# 主程序
image_path = r'C:\Users\Alan\Dropbox\1\output\201906180320.png'
process_and_analyze_image(image_path)