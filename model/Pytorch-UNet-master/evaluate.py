import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float32)

            mask_pred = net(image)
            
            # 調整 mask_pred 的形狀
            mask_pred = mask_pred.squeeze(1)  # 移除通道維度
            
            assert net.n_classes == 1, 'This implementation expects binary segmentation'
            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            
            mask_pred = torch.sigmoid(mask_pred)
            
            print(f"Pred shape: {mask_pred.shape}, True shape: {mask_true.shape}")
            
            assert mask_pred.shape == mask_true.shape, f"Shape mismatch: pred {mask_pred.shape}, true {mask_true.shape}"
            
            dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)