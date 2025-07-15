# Photo Enhancer

AI-powered photo enhancement using Real-ESRGAN and GFPGAN, optimized for RTX 3070 Ti.

## Setup

1. **Install CUDA toolkit:**
   ```bash
   sudo apt update && sudo apt upgrade -y
   
   # Add NVIDIA CUDA repository and key
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
   sudo dpkg -i cuda-keyring_1.0-1_all.deb
   sudo apt update
   
   # Install CUDA Toolkit 11.8 (more stable than 12.8)
   sudo apt install cuda-toolkit-11-8
   
   # Add CUDA to PATH
   echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **Verify CUDA installation:**
   ```bash
   nvcc --version
   nvidia-smi
   ```

3. **Create virtual environment with Python 3.11:**
   ```bash
   # Use Python 3.11 for better compatibility
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

4. **Upgrade build tools (CRITICAL STEP):**
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

5. **Install PyTorch with CUDA 11.8 support:**
   ```bash
   pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```

6. **Install AI enhancement libraries:**
   ```bash
   pip install realesrgan gfpgan basicsr facexlib opencv-python
   pip install -r requirements.txt
   ```

7. **Verify complete setup:**
   ```bash
   python -c "
   import torch
   from realesrgan import RealESRGANer
   from gfpgan import GFPGANer
   print(f'✓ PyTorch: {torch.__version__}')
   print(f'✓ CUDA available: {torch.cuda.is_available()}')
   print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
   print('✓ All libraries imported successfully!')
   "
   ```


## Usage

1. **Add your photos to `input_photos/` directory**

2. **Run enhancement:**
   ```bash
   python enhance_photos.py
   ```

3. **Enhanced photos will be saved to `enhanced_photos/`**

## Options

- `--input DIR`: Input directory (default: input_photos)
- `--output DIR`: Output directory (default: enhanced_photos)  
- `--scale N`: Upscale factor 2 or 4 (default: 4)
- `--no-face`: Disable face enhancement

## Examples

```bash
# Basic enhancement
python enhance_photos.py

# Custom directories
python enhance_photos.py --input my_photos --output results

# 2x upscale only
python enhance_photos.py --scale 2 --no-face
```

## Features

- **Real-ESRGAN**: General image super-resolution
- **GFPGAN**: Specialized face enhancement for graduation photos
- **CUDA acceleration**: Optimized for RTX 3070 Ti (8GB VRAM)
- **Memory management**: Automatic tiling for large images
- **Batch processing**: Process multiple photos automatically

## Troubleshooting

**If you get compatibility errors with Python 3.12:**
- Use Python 3.11 instead: `python3.11 -m venv .venv`
- The AI libraries have better compatibility with Python 3.11

**If PyTorch 2.5.x causes issues:**
- Use the stable PyTorch 2.0.1 with CUDA 11.8 as shown above
- This combination is well-tested and stable

**If you get build errors during installation:**
```bash
# The most common fix - upgrade build tools first
pip install --upgrade pip setuptools wheel
```

**If CUDA is not detected:**
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**If models fail to download:**
```bash
# Manually download models
mkdir -p models
cd models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
cd ..
```

## Project Structure
```
photo_enhancer/
├── interactive_enhance.py # PyCharm interactive script
├── README.md             # This file
├── requirements.txt      # Tested package versions
├── input_photos/         # Put your photos here
├── enhanced_photos/      # Enhanced results
├── models/              # Downloaded AI models
└── .venv/               # Virtual environment
```
## Performance Tips

- **RTX 3070 Ti (8GB)**: Use default settings
- **Lower VRAM GPUs**: Reduce tile size in script
- **CPU only**: Remove --index-url from PyTorch install
- **Large photos**: Script automatically handles tiling

## Working Configuration

This setup has been tested and verified working:
- **OS**: WSL2 Ubuntu 24.04
- **Python**: 3.11.13
- **PyTorch**: 2.0.1+cu118
- **CUDA**: 11.8
- **GPU**: RTX 3070 Laptop GPU (8GB)
