#!/usr/bin/env python3
"""
Interactive Photo Enhancement Script
Use # %% to create cells that can be run independently in PyCharm
"""

# %% Setup and imports
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
import time
import urllib.request
import os

print("=== GPU Setup Info ===")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
else:
    print("Using CPU (will be slower)")

# %% Configuration
INPUT_DIR = "input_photos"
OUTPUT_DIR = "enhanced_photos"
MODELS_DIR = "models"
SCALE = 4
USE_FACE_ENHANCEMENT = True  # Set to False to disable GFPGAN

# Create directories if they don't exist
Path(INPUT_DIR).mkdir(exist_ok=True)
Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(MODELS_DIR).mkdir(exist_ok=True)

print(f"Input directory: {INPUT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Models directory: {MODELS_DIR}")


# %% Download required models
def download_models():
    """Download AI models if they don't exist"""
    models = {
        "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    }

    for model_name, url in models.items():
        model_path = Path(MODELS_DIR) / model_name
        if not model_path.exists():
            print(f"Downloading {model_name}...")
            try:
                urllib.request.urlretrieve(url, model_path)
                print(f"✓ Downloaded {model_name}")
            except Exception as e:
                print(f"✗ Failed to download {model_name}: {e}")
        else:
            print(f"✓ {model_name} already exists")


# Download models
download_models()


# %% Initialize enhancers
def setup_enhancers():
    """Initialize Real-ESRGAN and GFPGAN enhancers"""

    # Real-ESRGAN setup
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=SCALE)

    realesrgan_model_path = Path(MODELS_DIR) / "RealESRGAN_x4plus.pth"

    upsampler = RealESRGANer(
        scale=SCALE,
        model_path=str(realesrgan_model_path),
        model=model,
        tile=400,  # Optimized for RTX 3070 Ti 8GB
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),  # Use FP16 for better performance
        gpu_id=0 if torch.cuda.is_available() else None
    )

    # GFPGAN setup for face enhancement
    face_enhancer = None
    if USE_FACE_ENHANCEMENT:
        gfpgan_model_path = Path(MODELS_DIR) / "GFPGANv1.4.pth"
        if gfpgan_model_path.exists():
            try:
                face_enhancer = GFPGANer(
                    model_path=str(gfpgan_model_path),
                    upscale=2,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=upsampler  # Use Real-ESRGAN for background
                )
                print("✓ GFPGAN face enhancer loaded")
            except Exception as e:
                print(f"✗ Could not load GFPGAN: {e}")
                print("Will use Real-ESRGAN only")
        else:
            print("✗ GFPGAN model not found, using Real-ESRGAN only")

    return upsampler, face_enhancer


# Initialize enhancers
print("\n=== Loading AI Models ===")
upsampler, face_enhancer = setup_enhancers()
print("✓ Real-ESRGAN loaded successfully!")


# %% Load and preview images
def load_image(filename):
    """Load an image file"""
    img_path = Path(INPUT_DIR) / filename
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not load {img_path}")
    return img


def display_image_info(img, title="Image"):
    """Display image information"""
    h, w, c = img.shape
    print(f"{title} - Size: {w}x{h}, Channels: {c}")
    print(f"Memory usage: ~{(h * w * c * 4) / 1024 / 1024:.1f} MB")


# List available images
input_path = Path(INPUT_DIR)
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
available_images = []
for ext in image_extensions:
    available_images.extend([f.name for f in input_path.glob(ext)])
    available_images.extend([f.name for f in input_path.glob(ext.upper())])

print(f"\n=== Available Images ({len(available_images)}) ===")
for i, img_name in enumerate(available_images):
    print(f"{i + 1}. {img_name}")

if not available_images:
    print(f"⚠️  No images found in {INPUT_DIR}/")
    print("Please add some photos to enhance!")
else:
    # Load first image for preview
    current_image = load_image(available_images[0])
    display_image_info(current_image, "Original")

    # Display original image
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Original: {available_images[0]}")
    plt.axis('off')
    plt.show()


# %% Enhance single image
def enhance_single_image(img, use_face_enhancer=True):
    """Enhance a single image using Real-ESRGAN and optionally GFPGAN"""
    start_time = time.time()

    if face_enhancer and use_face_enhancer and USE_FACE_ENHANCEMENT:
        print("Enhancing with GFPGAN (face-optimized)...")
        try:
            # GFPGAN handles both face and background enhancement
            _, _, enhanced = face_enhancer.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )
        except Exception as e:
            print(f"GFPGAN failed: {e}")
            print("Falling back to Real-ESRGAN...")
            enhanced, _ = upsampler.enhance(img, outscale=SCALE)
    else:
        print("Enhancing with Real-ESRGAN...")
        enhanced, _ = upsampler.enhance(img, outscale=SCALE)

    processing_time = time.time() - start_time
    print(f"✓ Enhancement completed in {processing_time:.2f} seconds")
    display_image_info(enhanced, "Enhanced")

    return enhanced


# Run enhancement on current image
if 'current_image' in locals():
    print(f"\n=== Enhancing {available_images[0]} ===")
    enhanced_image = enhance_single_image(current_image)

    # Display side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Original
    ax1.imshow(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"Original\n{current_image.shape[1]}x{current_image.shape[0]}")
    ax1.axis('off')

    # Enhanced
    ax2.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"Enhanced x{SCALE}\n{enhanced_image.shape[1]}x{enhanced_image.shape[0]}")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


# %% Save enhanced image
def save_enhanced_image(enhanced_img, original_filename):
    """Save enhanced image with descriptive filename"""
    output_path = Path(OUTPUT_DIR)

    # Create descriptive filename
    stem = Path(original_filename).stem
    ext = Path(original_filename).suffix
    enhancement_type = "gfpgan" if face_enhancer and USE_FACE_ENHANCEMENT else "realesrgan"
    output_filename = f"enhanced_{stem}_x{SCALE}_{enhancement_type}{ext}"
    full_path = output_path / output_filename

    # Save with high quality
    if ext.lower() in ['.jpg', '.jpeg']:
        cv2.imwrite(str(full_path), enhanced_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(str(full_path), enhanced_img)

    print(f"💾 Saved: {full_path}")
    return full_path


# Save the current enhanced image
if 'enhanced_image' in locals() and available_images:
    saved_path = save_enhanced_image(enhanced_image, available_images[0])


# %% Batch process all images
def batch_enhance_all():
    """Process all images in the input directory"""
    if not available_images:
        print("No images to process!")
        return []

    print(f"\n=== Batch Processing {len(available_images)} Images ===")
    results = []

    for i, filename in enumerate(available_images, 1):
        print(f"\n[{i}/{len(available_images)}] Processing {filename}...")
        try:
            img = load_image(filename)
            enhanced = enhance_single_image(img)
            saved_path = save_enhanced_image(enhanced, filename)
            results.append(saved_path)
        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")

    print(f"\n🎉 Batch processing complete!")
    print(f"✓ Successfully enhanced {len(results)} images")
    print(f"📁 Results saved in: {OUTPUT_DIR}/")

    return results


# Uncomment the line below to process all images
batch_results = batch_enhance_all()

# %% Quality metrics and comparison
def calculate_enhancement_metrics(original, enhanced):
    """Calculate and display enhancement metrics"""
    print("\n=== Enhancement Metrics ===")

    # Size comparison
    orig_h, orig_w = original.shape[:2]
    enh_h, enh_w = enhanced.shape[:2]

    print(f"Original size: {orig_w}x{orig_h}")
    print(f"Enhanced size: {enh_w}x{enh_h}")
    print(f"Scale factor: {enh_w / orig_w:.1f}x")

    # Memory usage
    orig_pixels = orig_h * orig_w
    enh_pixels = enh_h * enh_w
    print(f"Pixel count increase: {enh_pixels / orig_pixels:.1f}x")
    print(f"Estimated file size increase: {enh_pixels / orig_pixels:.1f}x")

    # Basic quality metrics
    orig_mean = np.mean(original)
    enh_mean = np.mean(enhanced)
    print(f"Average brightness: {orig_mean:.1f} → {enh_mean:.1f}")


# Calculate metrics for current images
if 'current_image' in locals() and 'enhanced_image' in locals():
    calculate_enhancement_metrics(current_image, enhanced_image)


# %% Performance monitoring
def gpu_memory_info():
    """Display GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
        print(f"\n=== GPU Memory Usage ===")
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved: {reserved:.2f} GB")
        print(f"Free: {8.0 - reserved:.2f} GB")  # Assuming 8GB RTX 3070 Ti


# Check GPU memory usage
gpu_memory_info()

print("\n=== Script Complete ===")
print("✓ All functions loaded and ready to use")
print(f"📁 Input folder: {INPUT_DIR}/")
print(f"📁 Output folder: {OUTPUT_DIR}/")
print("\n💡 Tips:")
print("- Run cells individually to process specific images")
print("- Uncomment 'batch_results = batch_enhance_all()' to process all images")
print("- Adjust SCALE and USE_FACE_ENHANCEMENT variables as needed")
