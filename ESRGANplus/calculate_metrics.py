import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

def calculate_metrics(hr_path, sr_path):
    # Load images
    hr_img = cv2.imread(hr_path)
    sr_img = cv2.imread(sr_path)

    # Debug: Check if images loaded
    if hr_img is None:
        raise ValueError(f"HR image not loaded: {hr_path}")
    if sr_img is None:
        raise ValueError(f"SR image not loaded: {sr_path}")

    # Resize SR to match HR (if needed)
    if hr_img.shape != sr_img.shape:
        sr_img = cv2.resize(sr_img, (hr_img.shape[1], hr_img.shape[0]))

    # Convert to YCrCb and extract Y channel (for SSIM)
    hr_y = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    sr_y = cv2.cvtColor(sr_img, cv2.COLOR_BGR2YCR_CB)[:, :, 0]

    # Calculate PSNR and SSIM
    psnr = cv2.PSNR(hr_img, sr_img)
    ssim_val = ssim(hr_y, sr_y, data_range=255)

    return psnr, ssim_val

# Example usage (UPDATE THESE PATHS!)
# hr_path = r"E:\ESRGANplus-YT\ESRGANplus\test_image\results\HR\0830.png"
hr_path = "./hr"
# sr_path = r"E:\ESRGANplus-YT\ESRGANplus\test_image\results\0830.png"
sr_path = "./results"

# Verify paths
if not os.path.exists(hr_path):
    print(f"HR image missing: {hr_path}")
if not os.path.exists(sr_path):
    print(f"SR image missing: {sr_path}")

# Calculate metrics
try:
    psnr_val, ssim_val = calculate_metrics(hr_path, sr_path)
    print(f"PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
except Exception as e:
    print(f"Error: {e}")