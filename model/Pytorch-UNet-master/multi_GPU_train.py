import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split, DistributedSampler
from tqdm import tqdm
import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
import time
import psutil
import GPUtil
from torch.profiler import profile, record_function, ProfilerActivity
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

os.environ['WANDB_PROJECT'] = 'U-Net'
os.environ['WANDB_ENTITY'] = 'alanalanalan0807'

# 定义全局变量
dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_gpu_info(rank):
    return {
        f'GPU{rank}_util': torch.cuda.utilization(rank),
        f'GPU{rank}_mem': torch.cuda.memory_allocated(rank) / torch.cuda.get_device_properties(rank).total_memory * 100
    }

def plot_to_wandb(fig):
    return wandb.Image(fig)

def visualize_predictions(images, true_masks, pred_masks, num_samples=5):
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    for i in range(num_samples):
        axes[i, 0].imshow(images[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 0].set_title("Input Image")
        axes[i, 1].imshow(true_masks[i].cpu().numpy(), cmap='gray')
        axes[i, 1].set_title("True Mask")
        axes[i, 2].imshow(pred_masks[i].cpu().numpy() > 0.5, cmap='gray')
        axes[i, 2].set_title("Predicted Mask")
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten() > 0.5)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

def plot_lr_curve(optimizer):
    lrs = optimizer.param_groups[0]['lr']
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.title('Learning Rate over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    return plt.gcf()

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
    plt.figure(figsize=(10, 7))
    plt.bar(range(len(ave_grads)), ave_grads, align="center")
    plt.xticks(range(len(ave_grads)), layers, rotation="vertical")
    plt.title("Gradient Flow")
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.tight_layout()
    return plt.gcf()

def train_model(
        rank,
        world_size,
        model,
        dir_img,
        dir_mask,
        dir_checkpoint,
        epochs: int = 50,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    dataset = BasicDataset(dir_img, dir_mask, img_scale, mask_suffix='_mask')
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank)
    
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val_set, sampler=val_sampler, **loader_args)

    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    if rank == 0:
        try:
            experiment = wandb.init(project='U-Net', entity='alanalanalan0807', name='enhanced_training_run')
            experiment.config.update(
                dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                     val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
            )
        except Exception as e:
            print(f"Could not log in to wandb. Error: {e}")
            experiment = None

    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.BCEWithLogitsLoss() if model.module.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img', disable=rank != 0) as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.module.n_channels, \
                    f'Network has been defined with {model.module.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.module.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.module.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                if rank == 0 and global_step % 50 == 0:  # 每50步记录一次详细信息
                    gpu_info = {f'GPU{i}_util': torch.cuda.utilization(i) for i in range(world_size)}
                    gpu_info.update({f'GPU{i}_mem': torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory * 100 for i in range(world_size)})
                    
                    # 预测可视化
                    with torch.no_grad():
                        pred_masks = torch.sigmoid(model(images))
                    vis_fig = visualize_predictions(images, true_masks, pred_masks)
                    
                    # 混淆矩阵
                    cm_fig = plot_confusion_matrix(true_masks.cpu().numpy(), pred_masks.cpu().numpy())
                    
                    # 梯度流
                    grad_fig = plot_grad_flow(model.named_parameters())
                    
                    if experiment:
                        experiment.log({
                            'train loss': loss.item(),
                            'step': global_step,
                            'epoch': epoch,
                            'learning_rate': optimizer.param_groups[0]['lr'],
                            'sample_predictions': plot_to_wandb(vis_fig),
                            'confusion_matrix': plot_to_wandb(cm_fig),
                            'gradient_flow': plot_to_wandb(grad_fig),
                            **gpu_info
                        })
                    else:
                        logging.info(f'Step {global_step}: loss: {loss.item():.4f}, GPU info: {gpu_info}')
                    
                    plt.close('all')  # 清理matplotlib图形

                pbar.set_postfix(**{'loss (batch)': loss.item(), **get_gpu_info(rank)})


        if rank == 0:
            val_score = evaluate(model, val_loader, device, amp)
            scheduler.step(val_score)

            # 学习率曲线
            lr_fig = plot_lr_curve(optimizer)

            logging.info('Validation Dice score: {}'.format(val_score))
            if experiment:
                experiment.log({
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'validation Dice': val_score,
                    'epoch': epoch,
                    'lr_curve': plot_to_wandb(lr_fig)
                })

            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.module.state_dict()
                state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')

    cleanup()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    return parser.parse_args()

def main():
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = UNet(n_channels=4, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    world_size = torch.cuda.device_count()
    mp.spawn(train_model,
             args=(world_size, model, dir_img, dir_mask, dir_checkpoint, 
                   args.epochs, args.batch_size, args.lr, args.val / 100, 
                   True, args.scale, args.amp),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    main()