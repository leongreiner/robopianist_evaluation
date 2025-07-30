#!/bin/bash
# Script to install CUDA-enabled PyTorch

set -e

echo "🔧 Fixing PyTorch CUDA installation..."
echo "Current date: $(date)"

# Setup conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pianist

echo "===== Current PyTorch Version ====="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo "===== Available CUDA Version ====="
# Load CUDA module like in the working test
module purge 2>/dev/null || true
module load cuda 2>/dev/null && echo "CUDA module loaded" || echo "Failed to load CUDA module"

# Show CUDA version
nvcc --version 2>/dev/null || echo "nvcc not available"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

echo "===== Installing CUDA-enabled PyTorch ====="
echo "Uninstalling current PyTorch..."
conda uninstall -y pytorch torchvision torchaudio

echo "Installing PyTorch with CUDA 12.1 support..."
# Install PyTorch with CUDA 12.1 (compatible with CUDA 12.6)
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

echo "===== Verifying Installation ====="
python -c "
import torch
print(f'New PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    print('✅ CUDA-enabled PyTorch successfully installed!')
else:
    print('❌ CUDA still not available after installation')
"

echo "===== PyTorch CUDA Fix Completed ====="
echo "You can now run your training scripts with GPU support!"
