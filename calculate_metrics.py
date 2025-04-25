import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchmetrics.classification import Precision, Recall
import lpips

# Paths to your images
real_path = r"C:\Users\scruk\pytorch-CycleGAN-and-pix2pix\results\model\test_latest\images\360_F_247028431_yPo8nwG9HuQN6oHyix8YnhYBeOXtF0c4_fake.png"
rec_path = r"C:\Users\scruk\pytorch-CycleGAN-and-pix2pix\results\model\test_latest\images\360_F_247028431_yPo8nwG9HuQN6oHyix8YnhYBeOXtF0c4_real.png"

# Load images
real_img = Image.open(real_path)
rec_img = Image.open(rec_path)

# Convert to numpy arrays for PSNR, SSIM, MSE
real_np = np.array(real_img)
rec_np = np.array(rec_img)

# Calculate PSNR
psnr_value = psnr(real_np, rec_np)

# Calculate SSIM with proper channel axis
ssim_value = ssim(real_np, rec_np, channel_axis=2, win_size=3)

# Calculate MSE
mse = np.mean((real_np - rec_np) ** 2)

# Calculate Precision and Recall with task parameter
precision_metric = Precision(task="multiclass", num_classes=2, average='macro')
recall_metric = Recall(task="multiclass", num_classes=2, average='macro')

# Assuming binary classification, convert images to binary
real_binary = (real_np > 128).astype(int)  # Threshold at 128
rec_binary = (rec_np > 128).astype(int)

# Update metrics
precision = precision_metric(torch.tensor(rec_binary), torch.tensor(real_binary))
recall = recall_metric(torch.tensor(rec_binary), torch.tensor(real_binary))

# Calculate LPIPS
lpips_fn = lpips.LPIPS(net='alex')
lpips_value = lpips_fn(transforms.ToTensor()(real_img).unsqueeze(0), transforms.ToTensor()(rec_img).unsqueeze(0))

print(f"LPIPS: {lpips_value.detach().item():.4f}")  # Added .detach() to prevent the warning
print(f"PSNR: {psnr_value:.2f} dB")
print(f"SSIM: {ssim_value:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"MSE: {mse:.2f}")
