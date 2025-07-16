import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def validate_directories(output_dir, gt_dir):
    """Validate that directories exist and contain images."""
    if not os.path.exists(gt_dir):
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
        
    gt_files = [f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    output_files = [f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not gt_files:
        raise ValueError(f"No images found in ground truth directory: {gt_dir}")
    if not output_files:
        raise ValueError(f"No images found in output directory: {output_dir}")
        
    return output_files, gt_files

def calculate_metrics(output_dir, gt_dir):
    """
    Calculate PSNR and SSIM between output images and ground truth images.
    
    Args:
        output_dir: Path to directory containing output images
        gt_dir: Path to directory containing ground truth images
        
    Returns:
        Dictionary with average PSNR and SSIM values
    """
    try:
        output_files, gt_files = validate_directories(output_dir, gt_dir)
        
        # Match files that exist in both directories
        common_files = set(output_files) & set(gt_files)
        if not common_files:
            raise ValueError("No matching image files found between output and GT directories")
            
        psnr_values = []
        ssim_values = []
        processed_files = []
        
        for filename in sorted(common_files):
            # Read images
            out_path = os.path.join(output_dir, filename)
            gt_path = os.path.join(gt_dir, filename)
            
            img_out = cv2.imread(out_path)
            img_gt = cv2.imread(gt_path)
            
            if img_out is None:
                print(f"Warning: Could not read output image: {out_path}")
                continue
            if img_gt is None:
                print(f"Warning: Could not read ground truth image: {gt_path}")
                continue
            
            # Make sure images are the same size
            if img_out.shape != img_gt.shape:
                print(f"Resizing {filename} to match GT dimensions")
                img_out = cv2.resize(img_out, (img_gt.shape[1], img_gt.shape[0]))
            
            # Convert to float32 for calculations
            img_out = img_out.astype(np.float32)
            img_gt = img_gt.astype(np.float32)
            
            # Calculate metrics
            current_psnr = psnr(img_gt, img_out, data_range=255)
            current_ssim = ssim(img_gt, img_out, data_range=255, channel_axis=2)
            
            psnr_values.append(current_psnr)
            ssim_values.append(current_ssim)
            processed_files.append(filename)
            
            print(f"{filename} - PSNR: {current_psnr:.2f} dB, SSIM: {current_ssim:.4f}")
        
        if not psnr_values:
            raise ValueError("No valid image pairs were processed")
            
        # Calculate averages
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        
        print("\n=== Final Results ===")
        print(f"Processed {len(processed_files)} image pairs")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        
        return {
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'psnr_values': psnr_values,
            'ssim_values': ssim_values,
            'processed_files': processed_files
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Set paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_directory = os.path.join(script_dir, "results")
    gt_directory = os.path.join(script_dir, "HR")
    
    print(f"Looking for ground truth images in: {gt_directory}")
    print(f"Looking for output images in: {output_directory}")
    
    metrics = calculate_metrics(output_directory, gt_directory)
    
    if metrics is None:
        print("\nFailed to calculate metrics. Please check:")
        print("1. Both 'HR' and 'results' folders exist")
        print("2. They contain matching image files with extensions .png, .jpg, or .jpeg")
        print("3. The images are readable (not corrupted)")