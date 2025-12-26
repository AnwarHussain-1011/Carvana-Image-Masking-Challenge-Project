import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net, full_img, device, scale_factor=1.0, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        # Resizing back to original image size for the final report
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().cpu().squeeze().numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    # Correctly points to your best training result
    parser.add_argument('--model', '-m', default='checkpoints/best_model.pth', help='Path to best_model.pth')
    parser.add_argument('--input', '-i', nargs='+', help='data/imgs', required=True)
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Scale factor (1.0 for best quality)')
    parser.add_argument('--viz', '-v', action='store_true', help='Visualize results')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initializing U-Net with 2 classes as per your successful training
    net = UNet(n_channels=3, n_classes=2, bilinear=False).to(device)
    logging.info(f'Loading model {args.model}')
    net.load_state_dict(torch.load(args.model, map_location=device))

    for fn in args.input:
        if not os.path.exists(fn):
            logging.error(f"File not found: {fn}")
            continue

        img = Image.open(fn)
        mask = predict_img(net, img, device, args.scale)
        
        # Save output to your outputs directory
        out_fn = f"outputs/{Path(fn).stem}_OUT.png"
        # Scaling by 255 to ensure the mask is visible in standard image viewers
        Image.fromarray((mask * 255).astype(np.uint8)).save(out_fn)
        logging.info(f'Saved prediction to {out_fn}')
        
        if args.viz:
            plot_img_and_mask(img, mask)