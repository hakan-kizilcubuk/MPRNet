"""
MPRNet Custom Inference and Evaluation Script
Supports: Denoising, Deblurring, and Deraining
Features: CPU/GPU dynamic device configuration, Single Image & Directory processing,
          PSNR/SSIM evaluation against ground truth, and side-by-side comparison saving.
"""

import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from runpy import run_path
from glob import glob
from collections import OrderedDict
from natsort import natsorted
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# Set seed for reproducibility (like DnCNN test suite)
torch.manual_seed(42)
np.random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="Custom Inference & Evaluation for MPRNet")
    parser.add_argument("--task", type=str, required=True, choices=["Deblurring", "Denoising", "Deraining"],
                        help="Restoration task to perform")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to input image or folder containing degraded/noisy images")
    parser.add_argument("--target_dir", type=str, default=None,
                        help="Optional path to target (ground truth) image or folder for metric calculation")
    parser.add_argument("--result_dir", type=str, default="./results/",
                        help="Directory to save the restored images")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to pretrained weights (.pth). Defaults to standard model paths.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device to run inference on ('cuda', 'cpu', or 'auto' for automatic detection)")
    parser.add_argument("--save_images", type=str, default="true", choices=["true", "false"],
                        help="Whether to save restored images")
    parser.add_argument("--save_comparison", type=str, default="false", choices=["true", "false"],
                        help="Whether to save side-by-side comparison images")
    parser.add_argument("--noise_level", type=float, default=None,
                        help="Optional noise level (sigma) to add synthetic Gaussian noise to inputs (for denoising tests)")
    return parser.parse_args()

def load_checkpoint(model, weights, device):
    checkpoint = torch.load(weights, map_location=device)
    state_dict_key = "state_dict" if "state_dict" in checkpoint else None
    
    # Extract state dict
    if state_dict_key:
        state_dict = checkpoint[state_dict_key]
    else:
        state_dict = checkpoint
        
    try:
        model.load_state_dict(state_dict)
    except Exception:
        # If model was saved with DataParallel, remove 'module.' prefix
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def save_img(filepath, img_rgb):
    """Save an RGB image using OpenCV, converting to BGR first."""
    cv2.imwrite(filepath, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

def compute_ssim(img1, img2):
    """Robust structural similarity calculation handling channel configurations."""
    # Input images are expected to be uint8 numpy arrays of shape (H, W, 3)
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        try:
            return compare_ssim(img1, img2, channel_axis=2, data_range=255)
        except TypeError:
            return compare_ssim(img1, img2, multichannel=True, data_range=255)
    else:
        return compare_ssim(img1, img2, data_range=255)

def main():
    args = parse_args()
    
    # 1. Device configuration
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\nUsing device: {device}")
    
    # 2. Gather image files
    if os.path.isfile(args.input_dir):
        input_files = [args.input_dir]
    else:
        raw_files = (
            glob(os.path.join(args.input_dir, "*.jpg")) +
            glob(os.path.join(args.input_dir, "*.JPG")) +
            glob(os.path.join(args.input_dir, "*.png")) +
            glob(os.path.join(args.input_dir, "*.PNG")) +
            glob(os.path.join(args.input_dir, "*.jpeg"))
        )
        # Deduplicate using set and normalize paths (important on Windows)
        input_files = natsorted(list(set(os.path.normpath(f) for f in raw_files)))
        
    if not input_files:
        raise FileNotFoundError(f"No valid image files found at: {args.input_dir}")
    print(f"Found {len(input_files)} input image(s) to process.")

    # 3. Handle model weights and import MPRNet
    weights_path = args.weights
    if not weights_path:
        weights_path = os.path.join(args.task, "pretrained_models", f"model_{args.task.lower()}.pth")
        
    print(f"Loading {args.task} model...")
    mprnet_module_path = os.path.join(args.task, "MPRNet.py")
    if not os.path.exists(mprnet_module_path):
        raise FileNotFoundError(f"Could not find model file at: {mprnet_module_path}")
        
    mprnet_module = run_path(mprnet_module_path)
    model = mprnet_module["MPRNet"]()
    
    if os.path.exists(weights_path):
        print(f"Loading checkpoint weights from: {weights_path}")
        load_checkpoint(model, weights_path, device)
    else:
        print(f"WARNING: Weights file not found at: {weights_path}. Model will run with random initialization.")
        
    model.to(device)
    model.eval()

    # 4. Create directories
    save_imgs = args.save_images == "true"
    save_comp = args.save_comparison == "true"
    
    if save_imgs:
        os.makedirs(args.result_dir, exist_ok=True)
    if save_comp:
        comp_dir = os.path.join(args.result_dir, "comparisons")
        os.makedirs(comp_dir, exist_ok=True)

    # Metrics accumulators
    psnrs = []
    ssims = []
    
    img_multiple_of = 8
    
    print("\nProcessing images...")
    for file_path in tqdm(input_files):
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Load and convert image to RGB
        img = Image.open(file_path).convert("RGB")
        input_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
        
        # Add synthetic noise if specified
        if args.noise_level is not None:
            noise = torch.randn_like(input_tensor) * (args.noise_level / 255.0)
            input_tensor = torch.clamp(input_tensor + noise, 0.0, 1.0)
            # Reconstruct the noisy image for saving/comparison
            noisy_np = img_as_ubyte(input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
            input_img_to_show = noisy_np
        else:
            input_img_to_show = np.array(img)
            
        # Get dimensions and pad if not multiple of 8
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        padh = (img_multiple_of - h % img_multiple_of) % img_multiple_of
        padw = (img_multiple_of - w % img_multiple_of) % img_multiple_of
        
        if padh > 0 or padw > 0:
            input_tensor = F.pad(input_tensor, (0, padw, 0, padh), "reflect")
            
        # Perform Inference
        with torch.no_grad():
            restored_tensor = model(input_tensor)
            
        # Extract main stage output (stage 3)
        restored_tensor = restored_tensor[0]
        restored_tensor = torch.clamp(restored_tensor, 0.0, 1.0)
        
        # Unpad output to original dimensions
        if padh > 0 or padw > 0:
            restored_tensor = restored_tensor[:, :, :h, :w]
            
        # Convert to numpy uint8
        restored_np = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        restored_img = img_as_ubyte(restored_np)
        
        # Save restored image
        if save_imgs:
            save_img(os.path.join(args.result_dir, f"{filename}.png"), restored_img)
            
        # Target (ground truth) comparison & evaluation
        target_img = None
        if args.target_dir:
            # Look for ground truth matching by filename
            target_path = None
            if os.path.isfile(args.target_dir):
                target_path = args.target_dir
            else:
                # Try finding same filename with any common extension
                for ext in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]:
                    potential_path = os.path.join(args.target_dir, filename + ext)
                    if os.path.exists(potential_path):
                        target_path = potential_path
                        break
            
            if target_path and os.path.exists(target_path):
                target_img = np.array(Image.open(target_path).convert("RGB"))
                
                # Calculate metrics
                psnr = compare_psnr(target_img, restored_img, data_range=255)
                ssim = compute_ssim(target_img, restored_img)
                psnrs.append(psnr)
                ssims.append(ssim)
                
                tqdm.write(f"Image: {filename} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")
            else:
                tqdm.write(f"WARNING: Target ground truth image for '{filename}' not found at: {args.target_dir}")

        # Save side-by-side comparison
        if save_comp:
            if target_img is not None:
                # Horizontal concat: Input | Restored | Target
                comparison = np.hstack((input_img_to_show, restored_img, target_img))
            else:
                # Horizontal concat: Input | Restored
                comparison = np.hstack((input_img_to_show, restored_img))
            
            save_img(os.path.join(comp_dir, f"{filename}_comparison.png"), comparison)

    # Print summary metrics
    if psnrs:
        mean_psnr = np.mean(psnrs)
        mean_ssim = np.mean(ssims)
        print("\n" + "="*50)
        print("Evaluation Summary")
        print(f"Processed: {len(psnrs)} images")
        print(f"Mean PSNR: {mean_psnr:.2f} dB")
        print(f"Mean SSIM: {mean_ssim:.4f}")
        print("="*50)
    else:
        print("\nInference completed. No ground truth metrics calculated.")
        
    if save_imgs:
        print(f"Restored images saved to: {args.result_dir}")
    if save_comp:
        print(f"Comparison images saved to: {comp_dir}")

if __name__ == "__main__":
    main()
