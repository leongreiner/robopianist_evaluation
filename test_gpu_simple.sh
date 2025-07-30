#!/bin/bash

echo "===== Simple GPU Test Script ====="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo ""

echo "===== System GPU Check ====="
echo "Checking for NVIDIA GPUs with lspci:"
lspci | grep -i nvidia || echo "No NVIDIA devices found"
echo ""

echo "===== GPU Device Files ====="
echo "Checking /dev/nvidia* files:"
ls -la /dev/nvidia* 2>/dev/null || echo "No /dev/nvidia* files found"
echo ""

echo "===== nvidia-smi Test ====="
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi found, running test:"
    nvidia-smi
else
    echo "nvidia-smi not found in PATH"
fi
echo ""

echo "===== Python/PyTorch GPU Test ====="
# Setup conda if available
if [ -f "/dss/dsshome1/0A/di97jur/miniconda3/bin/conda" ]; then
    echo "Setting up conda environment..."
    eval "$(/dss/dsshome1/0A/di97jur/miniconda3/bin/conda shell.bash hook)"
fi

python -c "
import sys
print('Python executable:', sys.executable)

try:
    import torch
    print('PyTorch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA version:', torch.version.cuda)
        print('Number of GPUs:', torch.cuda.device_count())
        print('GPU name:', torch.cuda.get_device_name(0))
    else:
        print('CUDA not available to PyTorch')
except ImportError:
    print('PyTorch not installed')
except Exception as e:
    print('PyTorch error:', e)
"

echo ""
echo "===== Test Completed ====="
