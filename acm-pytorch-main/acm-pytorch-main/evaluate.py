import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from model.segmentation import ASKCResNetFPN, ASKCResUNet
from utils.data import SirstDataset

# ======== Configuration Defaults ========
DEFAULT_BACKBONE     = 'UNet'
DEFAULT_FUSE_MODE    = 'AsymBi'
DEFAULT_DATASET_ROOT = './sirst'
DEFAULT_CHECKPOINT   = "./params/UNet_AsymBi/Epoch-225_IoU-0.7152_nIoU-0.7281.pkl"
DEFAULT_BATCH_SIZE   = 1
DEFAULT_PARALLEL     = False

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate ACM model')
    p.add_argument('--crop_size',     type=int, default=480)
    p.add_argument('--base_size',     type=int, default=512)
    p.add_argument('--batch_size',    type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument('--parallel',      type=int, choices=[0,1], default=int(DEFAULT_PARALLEL))
    p.add_argument('--backbone_mode', type=str, default=DEFAULT_BACKBONE, choices=['FPN','UNet'])
    p.add_argument('--fuse_mode',     type=str, default=DEFAULT_FUSE_MODE,   choices=['BiLocal','AsymBi','BiGlobal'])
    p.add_argument('--blocks_per_layer', type=int, default=4)
    p.add_argument('--dataset_root',  type=str, default=DEFAULT_DATASET_ROOT)
    p.add_argument('--checkpoint',    type=str, default=DEFAULT_CHECKPOINT)
    return p.parse_args()

def compute_iou_f1(pred_mask, true_mask):
    # both boolean arrays
    inter = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou  = inter / union if union > 0 else 0.0
    # F1 = 2TP / (2TP + FP + FN)
    tp = inter
    fp = np.logical_and(pred_mask, ~true_mask).sum()
    fn = np.logical_and(~pred_mask, true_mask).sum()
    f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn)>0 else 0.0
    return iou, f1

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- load checkpoint (full model or state_dict) ---
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, nn.Module):
        net = ckpt
    else:
        state_dict = ckpt
        blocks   = [args.blocks_per_layer]*3
        channels = [8,16,32,64]
        if args.backbone_mode=='FPN':
            net = ASKCResNetFPN(blocks, channels, args.fuse_mode)
        else:
            net = ASKCResUNet(blocks, channels, args.fuse_mode)
        net.load_state_dict(state_dict)

    net.to(device)
    if args.parallel:
        net = nn.DataParallel(net)
    net.eval()

    # --- data ---
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    ])
    test_ds = SirstDataset(args, mode='val')
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # --- inference + metrics ---
    out_dir = './test_results'
    os.makedirs(out_dir, exist_ok=True)
    ious, f1s = [], []

    print(f"Saving results to {out_dir}/")
    for idx, (img, mask) in enumerate(tqdm(test_loader, desc='Evaluating')):
        img = img.to(device)
        with torch.no_grad():
            out = net(img)

        # convert to binary mask
        pred = out.squeeze(0).cpu().numpy()
        pred_bin = (pred > 0.5).astype(np.uint8)  # threshold at 0.5
        true_bin = (mask.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)

        # compute metrics
        iou, f1 = compute_iou_f1(pred_bin, true_bin)
        ious.append(iou)
        f1s.append(f1)

        # save visual for sanity (optional)
        vis = (pred_bin * 255).astype(np.uint8).transpose(1,2,0)
        cv2.imwrite(f"{out_dir}/{idx:03d}_pred.png", vis)

    # --- print overall ---
    mean_iou = np.mean(ious)
    mean_f1  = np.mean(f1s)
    print(f"\nMean IoU: {mean_iou:.4f}")
    print(f"Mean F1:  {mean_f1:.4f}")

    # --- plot curves ---
    plt.figure()
    plt.plot(ious, label='IoU')
    plt.plot(f1s,  label='F1')
    plt.xlabel('Image index')
    plt.ylabel('Score')
    plt.title('Per-image IoU & F1 on Validation Set')
    plt.legend()
    plt.tight_layout()
    plt.savefig('metrics_plot.png')
    print("Saved plot to metrics_plot.png")

if __name__=='__main__':
    main()
