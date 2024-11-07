# Himawari8 Internal Wave Detection

使用圖像分割方法來偵測和分析海洋內波。這個專案主要處理向日葵8號衛星的影像數據。

## 專案結構
📁 image_processing/
├── 📁 cloud/                 # 雲層相關處理
├── 📁 model/                 # 模型定義和訓練
├── 📁 Multibands/           # 多波段影像處理
├── 📄 Image_Processing.ipynb # 主要的影像處理流程
├── 📄 Image_Enhancement.py   # 影像增強相關函數
├── 📄 Modified_Image_Processing_Pipeline.py # 改進的影像處理管線
├── 📄 NLM_MSR.py            # 非局部均值和多尺度視網膜演算法
├── 📄 Optical_flow.py       # 光流法計算
└── 📄 GaborFilter.py        # Gabor 濾波器實現

## 功能模組

### 前處理
- `check_training_datas.py`: 檢查訓練數據
- `data_augmentation.py`: 數據增強
- `labelmem2mask.py`: 標籤轉換為遮罩
- `move_masks.py`: 遮罩文件處理

### 特徵提取
- `Multibands/`: 多波段特徵提取
- `NLM_MSR.py`: 非局部均值和多尺度視網膜增強
- `calculate_IW_numbers.py`: 內波特徵計算
- `tryband.py`: 波段試驗

### 後處理
- `combine_picture.py`: 圖像合併
- `compare_json_png.py`: JSON 和 PNG 文件比較
- `remove_aug_imgs.py`: 移除增強圖像
- `movetxt.py`: 文本文件處理

### 可視化
- `plotGMT/`: GMT 繪圖工具
- `plotGMT_Gamma.py`: Gamma 校正可視化
- `video.py`: 視頻生成

## 環境配置
```bash
# 配置環境（建議使用 conda）
conda create -n wave python=3.8
conda activate wave
