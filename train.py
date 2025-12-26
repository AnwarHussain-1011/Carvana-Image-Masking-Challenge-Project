import argparse
import logging
import os
import time
import csv
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Local imports
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from confusion_matrix import generate_all_plots

# Directories setup
dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')
dir_outputs = Path('./outputs/')

dir_checkpoint.mkdir(parents=True, exist_ok=True)
(dir_outputs / 'plots').mkdir(parents=True, exist_ok=True)

def train_model(model, device, epochs=100, batch_size=4, learning_rate=1e-4, val_percent=0.1, img_scale=0.5, amp=True):
    log_file = dir_outputs / 'training_log.csv'
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_dice', 'learning_rate', 'epoch_time'])

    # 1. Dataset Loading
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except:
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    n_val = int(len(dataset) * val_percent)
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(42))
    
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count() or 1, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    # 2. Optimized Research Hyperparameters
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    grad_scaler = torch.amp.GradScaler('cuda', enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    best_val_dice = 0.0

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0
        
        with tqdm(total=len(train_set), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'].to(device), batch['mask'].to(device)

                # STABILITY FIX: Clamp labels to prevent CUDA assertion error
                true_masks = torch.clamp(true_masks, 0, model.n_classes - 1)

                with torch.autocast('cuda', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(torch.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(F.softmax(masks_pred, dim=1).float(), 
                                          F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(), 
                                          multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                epoch_loss += loss.item()

        # 4. Evaluation and Best Model Saving
        val_score = evaluate(model, val_loader, device, amp)
        scheduler.step() 
        
        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        logging.info(f'Epoch {epoch}: Loss {avg_loss:.4f}, Val Dice {val_score:.4f}')

        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, avg_loss, val_score, optimizer.param_groups[0]['lr'], epoch_time])

        if val_score > best_val_dice:
            best_val_dice = val_score
            torch.save(model.state_dict(), str(dir_checkpoint / 'best_model.pth'))
            logging.info(f'🚩 Best model updated: {best_val_dice:.4f}')

    # Generate Visualizations
    model.load_state_dict(torch.load(str(dir_checkpoint / 'best_model.pth')))
    generate_all_plots(log_file, model, val_loader, device, dir_outputs / 'plots')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize U-Net
    model = UNet(n_channels=3, n_classes=2, bilinear=False).to(device)
    
    train_model(model, device, epochs=100, batch_size=4, learning_rate=1e-4, amp=True)