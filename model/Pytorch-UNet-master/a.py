import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
import torch
print(f"GPU 數量: {torch.cuda.device_count()}")
print(f"當前 GPU 名稱: {torch.cuda.get_device_name(0)}")
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)