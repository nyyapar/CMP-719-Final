import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import OrderedDict

from models.discriminator import Discriminator
from models.generator1_can8 import Generator1_CAN8
from models.generator2_ucan64 import Generator2_UCAN64
from tools.dataloader import SirstDataset
from tools.log import initialize_logger
from tools.fmeasure import calculateF1Measure

# Configuration
TEST_BATCH_SIZE = 1
USE_PARALLEL   = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize logger (if needed)
initialize_logger('./logs')

# Prepare transforms
composed = transforms.Compose([
    transforms.Grayscale(1),
    transforms.ToTensor(),
])

# Create output folder
os.makedirs('test_results', exist_ok=True)

# Dataset and loader
test_dataset = SirstDataset(mode='val', crop_size=256, base_size=256, root='./sirst')
test_loader  = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
# Use dataset.names for output filenames
filenames = test_dataset.names

# Build and load models
disc_net = Discriminator().to(DEVICE)
gen1_net = Generator1_CAN8().to(DEVICE)
gen2_net = Generator2_UCAN64().to(DEVICE)

if USE_PARALLEL and torch.cuda.is_available():
    disc_net = nn.DataParallel(disc_net)
    gen1_net = nn.DataParallel(gen1_net)
    gen2_net = nn.DataParallel(gen2_net)

# Helper to strip DataParallel prefixes
def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location=DEVICE)
    new_state = OrderedDict()
    for k, v in checkpoint.items():
        name = k.replace('module.', '')
        new_state[name] = v
    model.load_state_dict(new_state)

# Load pretrained weights (adjust epoch number)
load_checkpoint(disc_net, './saved_models/discriminator_epoch_9.pth')
load_checkpoint(gen1_net, './saved_models/generator1_epoch_9.pth')
load_checkpoint(gen2_net, './saved_models/generator2_epoch_9.pth')

# Set eval mode
disc_net.eval()
gen1_net.eval()
gen2_net.eval()

# Initialize metric accumulators
sum_iou1 = sum_f1_1 = 0.0
sum_iou2 = sum_f1_2 = 0.0
sum_iou3 = sum_f1_3 = 0.0

# Inference and metrics
with torch.no_grad():
    for ix, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):
        img  = x.to(DEVICE)
        mask = y.to(DEVICE)  # shape (1, H, W)
        gt = (mask.squeeze(0).cpu().numpy() > 0).astype(np.uint8)

        # G1
        out1 = gen1_net(img)[0].cpu().numpy()  # shape (1, H, W)
        pred1 = (out1[0] > 0.5).astype(np.uint8)
        inter1 = np.logical_and(pred1, gt).sum()
        union1 = np.logical_or(pred1, gt).sum()
        iou1 = inter1 / (union1 + 1e-6)
        f1_1 = calculateF1Measure(pred1, gt, 0.5)
        sum_iou1 += iou1
        sum_f1_1 += f1_1
        # save G1 mask
        mask1 = (pred1 * 255).astype(np.uint8)
        cv2.imwrite(f'test_results/{filenames[ix]}_G1.png', mask1)

        # G2
        out2 = gen2_net(img)[0].cpu().numpy()
        pred2 = (out2[0] > 0.5).astype(np.uint8)
        inter2 = np.logical_and(pred2, gt).sum()
        union2 = np.logical_or(pred2, gt).sum()
        iou2 = inter2 / (union2 + 1e-6)
        f1_2 = calculateF1Measure(pred2, gt, 0.5)
        sum_iou2 += iou2
        sum_f1_2 += f1_2
        mask2 = (pred2 * 255).astype(np.uint8)
        cv2.imwrite(f'test_results/{filenames[ix]}_G2.png', mask2)

        # Fusion
        fusion = (out1 + out2) / 2  # shape (1,H,W)
        pred3 = (fusion[0] > 0.5).astype(np.uint8)
        inter3 = np.logical_and(pred3, gt).sum()
        union3 = np.logical_or(pred3, gt).sum()
        iou3 = inter3 / (union3 + 1e-6)
        f1_3 = calculateF1Measure(pred3, gt, 0.5)
        sum_iou3 += iou3
        sum_f1_3 += f1_3
        mask3 = (pred3 * 255).astype(np.uint8)
        cv2.imwrite(f'test_results/{filenames[ix]}_Res.png', mask3)

# Print average metrics
n = len(test_loader)
print(f"G1 - mean IoU: {sum_iou1/n:.4f}, mean F1: {sum_f1_1/n:.4f}")
print(f"G2 - mean IoU: {sum_iou2/n:.4f}, mean F1: {sum_f1_2/n:.4f}")
print(f"Fusion - mean IoU: {sum_iou3/n:.4f}, mean F1: {sum_f1_3/n:.4f}")
