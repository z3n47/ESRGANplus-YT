import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(results_dir, hr_dir):
    """Calculate PSNR and SSIM between enhanced and ground truth images"""
    print(f"Checking directories:\nResults: {results_dir}\nHR: {hr_dir}")
    
    # Get list of image files
    result_files = sorted([f for f in os.listdir(results_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    hr_files = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"\nFound {len(result_files)} result files and {len(hr_files)} HR files")
    
    if not result_files:
        print(f"Error: No images found in results directory: {results_dir}")
        return [], []
    if not hr_files:
        print(f"Error: No images found in HR directory: {hr_dir}")
        return [], []
    
    psnr_values = []
    ssim_values = []
    
    for res_file in result_files:
        # Find matching HR file (remove possible suffixes added during processing)
        base_name = os.path.splitext(res_file)[0]
        hr_file = next((f for f in hr_files if os.path.splitext(f)[0] in base_name), None)
        
        if not hr_file:
            print(f"No matching HR file found for {res_file}")
            continue
            
        print(f"\nProcessing {res_file} vs {hr_file}")
        
        # Read images
        res_path = os.path.join(results_dir, res_file)
        hr_path = os.path.join(hr_dir, hr_file)
        
        img_res = cv2.imread(res_path)
        img_hr = cv2.imread(hr_path)
        
        if img_res is None:
            print(f"Warning: Could not read result image {res_path}")
            continue
        if img_hr is None:
            print(f"Warning: Could not read HR image {hr_path}")
            continue
        
        # Make sure images are the same size
        if img_res.shape != img_hr.shape:
            print(f"Resizing {res_file} to match {hr_file} dimensions")
            img_res = cv2.resize(img_res, (img_hr.shape[1], img_hr.shape[0]))
        
        # Convert to float32 and scale to 0-1 range
        img_res = img_res.astype(np.float32) / 255.0
        img_hr = img_hr.astype(np.float32) / 255.0
        
        # Calculate metrics
        try:
            current_psnr = psnr(img_hr, img_res, data_range=1.0)
            current_ssim = ssim(img_hr, img_res, data_range=1.0, channel_axis=2)
            
            psnr_values.append(current_psnr)
            ssim_values.append(current_ssim)
            
            print(f"PSNR: {current_psnr:.4f} dB, SSIM: {current_ssim:.4f}")
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            continue
    
    if psnr_values and ssim_values:
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        print("\nAverage Metrics:")
        print(f"PSNR: {avg_psnr:.4f} dB")
        print(f"SSIM: {avg_ssim:.4f}")
    else:
        print("\nNo valid metric values were calculated.")
    
    return psnr_values, ssim_values

if __name__ == "__main__":
    # Use absolute paths to be sure
    project_root = os.path.dirname(os.path.abspath(__file__))
    results_directory = os.path.join(project_root, "test_image", "results")
    hr_directory = os.path.join(project_root, "test_image", "HR")
    
    print(f"Project root: {project_root}")
    print(f"Results directory: {results_directory}")
    print(f"HR directory: {hr_directory}")
    
    if not os.path.exists(results_directory):
        print(f"\nError: Results directory not found at {results_directory}")
        print("Trying alternative path...")
        # Try alternative path if the first one fails
        results_directory = os.path.join(project_root, "ESRGANplus", "results")
        print(f"Trying: {results_directory}")
    
    if not os.path.exists(hr_directory):
        print(f"\nError: HR directory not found at {hr_directory}")
        print("Trying alternative path...")
        hr_directory = os.path.join(project_root, "ESRGANplus", "test_image", "HR")
        print(f"Trying: {hr_directory}")
    
    if os.path.exists(results_directory) and os.path.exists(hr_directory):
        psnr_values, ssim_values = calculate_metrics(results_directory, hr_directory)
        
        # Save results to file
        with open(os.path.join(project_root, "metrics_results.txt"), "w") as f:
            f.write("PSNR,SSIM\n")
            for p, s in zip(psnr_values, ssim_values):
                f.write(f"{p:.4f},{s:.4f}\n")
            if psnr_values:
                f.write(f"\nAverage PSNR: {np.mean(psnr_values):.4f} dB\n")
                f.write(f"Average SSIM: {np.mean(ssim_values):.4f}\n")
        print("\nMetrics saved to metrics_results.txt")
    else:
        print("\nCannot proceed - directories not found")
        print("Please check:")
        print(f"1. Results directory exists: {os.path.exists(results_directory)}")
        print(f"2. HR directory exists: {os.path.exists(hr_directory)}")
        print("\nCurrent directory contents:")
        print(os.listdir(project_root))