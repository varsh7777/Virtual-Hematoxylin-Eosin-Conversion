from PIL import Image
import numpy as np
import cv2
import os
import math

img_dir = r"C:/Users/sriva/Downloads/fibi image snr test 900mAillum 85msecexp/"
print(img_dir)
extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

image_files = [
    f for f in os.listdir(img_dir)
    if f.lower().endswith(extensions)
][:100]

print(f"Processing {len(image_files)} images...\n")

gray_stack = []

for fname in image_files:
    img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_stack.append(gray)

gray_stack = np.stack(gray_stack, axis=0)
print("stack shape:", gray_stack.shape)

mean_signal = gray_stack.mean(axis=0)   # (H, W)
std_noise   = gray_stack.std(axis=0)    # (H, W)
eps = 1e-6
snr_map = mean_signal / (std_noise + eps)

# ---- single-number summaries ---- #
mean_signal_global = float(mean_signal.mean())
mean_noise_global  = float(std_noise.mean())

snr = mean_signal_global / mean_noise_global
snr_db = 20 * math.log10(snr)

print(f"Per-pixel averaged SNR: {snr:.2f}")
print(f"Per-pixel averaged SNR (dB): {snr_db:.2f}")
print(f"Mean noise * 2: {20 * math.log10(mean_noise_global) * 2:.2f}")
print(f"1 SD above: {20 * math.log10(snr + mean_noise_global):.2f}")
print(f"1 SD below: {20 * math.log10(snr - mean_noise_global):.2f}")