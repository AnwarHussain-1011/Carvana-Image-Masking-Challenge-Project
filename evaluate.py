import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            
            # FIX: Clamp labels to prevent CUDA assertion errors
            mask_true = torch.clamp(mask_true, 0, net.n_classes - 1)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                # convert to one-hot format for precise multiclass calculation
                mask_true_oh = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred_oh = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background (index 0)
                dice_score += multiclass_dice_coeff(mask_pred_oh[:, 1:], mask_true_oh[:, 1:], reduce_batch_first=False)

    net.train()
    return (dice_score / max(num_val_batches, 1)).item()