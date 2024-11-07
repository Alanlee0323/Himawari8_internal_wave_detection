import cv2
import numpy as np
import matplotlib.pyplot as plt

# 數據讀取函數
def read_data(file_path):
    data = np.loadtxt(file_path)
    return data

def analyze_edge_orientation(image, cell_size=(8, 8), bins=9):
    # Gabor 濾波器
    gabor_responses = []
    best_response = None
    best_theta = None
    max_response = 0

    for theta in range(80, 110, 2):  # 主要關注接近水平的方向
        for sigma in [3, 5, 7]:  # 調整高斯包絡的大小
            for lambd in [10, 15, 20]:  # 調整正弦波的波長
                kernel = cv2.getGaborKernel((31, 31), sigma, theta, lambd, 0.5, 0, ktype=cv2.CV_32F)
                gabor_response = cv2.filter2D(image, cv2.CV_32F, kernel)
                response_strength = np.sum(gabor_response)
                
                if response_strength > max_response:
                    max_response = response_strength
                    best_response = gabor_response
                    best_theta = theta

    return {
        'best_gabor_response': best_response,
        'best_theta': best_theta
    }

def visualize_orientation_analysis(image, analysis_results):
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('原始圖像')
    
    if analysis_results['best_gabor_response'] is not None:
        # 將 Gabor 響應正規化到 0-255 範圍
        gabor_normalized = cv2.normalize(analysis_results['best_gabor_response'], None, 0, 255, cv2.NORM_MINMAX)
        axs[1].imshow(gabor_normalized, cmap='gray')
        axs[1].set_title(f'最佳 Gabor 響應 (角度: {analysis_results["best_theta"]}°)')
    
    plt.tight_layout()
    plt.show()


def process_and_analyze_image(image_path):
    # 讀取圖像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"無法讀取圖像：{image_path}")
        return
    
    analysis_results = analyze_edge_orientation(image)
    visualize_orientation_analysis(image, analysis_results)
    
    print(f"最佳 Gabor 濾波器角度：{analysis_results['best_theta']} 度")

# 主程序
image_path = r'C:\Users\Alan\Dropbox\Himawari8_Wave_Speed_Detection\output5\201906170850.png'
process_and_analyze_image(image_path)