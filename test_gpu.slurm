#!/bin/sh
#SBATCH --job-name=test_gpu
#SBATCH --time=0:30:00
#SBATCH -o /dss/dsstbyfs02/pn52ru/pn52ru-dss-0000/di97jur/robopianist_evaluation/job_logs/test_gpu.%j.%N.out
#SBATCH -e /dss/dsstbyfs02/pn52ru/pn52ru-dss-0000/di97jur/robopianist_evaluation/job_logs/test_gpu.%j.%N.err
#SBATCH -D /dss/dsstbyfs02/pn52ru/pn52ru-dss-0000/di97jur/robopianist_evaluation/
#SBATCH --cluster=hpda2
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --partition=hpda2_compute_gpu

echo "===== GPU Test Job Info ====="
echo "Node List: " $SLURM_NODELIST
echo "jobID: " $SLURM_JOB_ID
echo "Partition: " $SLURM_JOB_PARTITION
echo "Submit directory:" $SLURM_SUBMIT_DIR
echo "Submit host:" $SLURM_SUBMIT_HOST
echo "In the directory: `pwd`"
echo "As the user: `whoami`"
echo "Date: `date`"
echo ""

# Create job logs directory
mkdir -p job_logs

echo "===== SLURM GPU Variables ====="
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "SLURM_GPU_BIND: $SLURM_GPU_BIND"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

echo "===== System Information ====="
echo "Hostname: `hostname`"
echo "Kernel: `uname -r`"
echo "OS: `cat /etc/os-release | grep PRETTY_NAME`"
echo ""

echo "===== Available Modules ====="
echo "Checking for CUDA modules..."
module avail 2>&1 | grep -i cuda || echo "No CUDA modules found"
echo ""
echo "Checking for GPU modules..."
module avail 2>&1 | grep -i gpu || echo "No GPU modules found"
echo ""
echo "Checking for driver modules..."
module avail 2>&1 | grep -i nvidia || echo "No NVIDIA modules found"
echo ""

echo "===== Loading Modules ====="
echo "Attempting to load CUDA modules..."
# Try common CUDA module names
for cuda_module in cuda cuda/12.2 cuda/11.8 cuda/12.1 CUDA; do
    echo "Trying to load: $cuda_module"
    if module load $cuda_module 2>/dev/null; then
        echo "Successfully loaded: $cuda_module"
        break
    else
        echo "Failed to load: $cuda_module"
    fi
done

# Try loading NVIDIA driver modules
for nvidia_module in nvidia nvidia-driver nvidia-drivers; do
    echo "Trying to load: $nvidia_module"
    if module load $nvidia_module 2>/dev/null; then
        echo "Successfully loaded: $nvidia_module"
        break
    else
        echo "Failed to load: $nvidia_module"
    fi
done

echo ""
echo "Currently loaded modules:"
module list 2>&1 || echo "Failed to list modules"
echo ""

echo "===== GPU Device Files ====="
echo "Checking /dev/nvidia* files:"
ls -la /dev/nvidia* 2>/dev/null || echo "No /dev/nvidia* files found"
echo ""
echo "Checking /proc/driver/nvidia:"
ls -la /proc/driver/nvidia/ 2>/dev/null || echo "No /proc/driver/nvidia found"
echo ""

echo "===== nvidia-smi Test ====="
echo "Testing nvidia-smi command..."
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi found in PATH"
    nvidia-smi
    smi_exit_code=$?
    echo "nvidia-smi exit code: $smi_exit_code"
else
    echo "nvidia-smi not found in PATH"
    echo "Searching for nvidia-smi..."
    find /usr -name "nvidia-smi" 2>/dev/null || echo "nvidia-smi not found in /usr"
fi
echo ""

echo "===== CUDA Compiler Test ====="
echo "Testing nvcc command..."
if command -v nvcc >/dev/null 2>&1; then
    echo "nvcc found in PATH"
    nvcc --version
else
    echo "nvcc not found in PATH"
    echo "Searching for nvcc..."
    find /usr -name "nvcc" 2>/dev/null || echo "nvcc not found in /usr"
fi
echo ""

echo "===== CUDA Runtime Test ====="
echo "Checking CUDA runtime libraries..."
ldconfig -p | grep cuda || echo "No CUDA libraries found in ldconfig"
echo ""
echo "Checking common CUDA library paths:"
for cuda_path in /usr/local/cuda /opt/cuda /usr/cuda; do
    if [ -d "$cuda_path" ]; then
        echo "Found CUDA directory: $cuda_path"
        ls -la $cuda_path/lib* 2>/dev/null || echo "No lib directories in $cuda_path"
    else
        echo "CUDA directory not found: $cuda_path"
    fi
done
echo ""

echo "===== Environment Variables ====="
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_PATH: $CUDA_PATH"
echo ""

echo "===== Python/Conda Setup ====="
module load cuda/11.8 2>&1 || echo "Failed to load cuda/11.8"
module load nvidia/cuda/11.8 2>&1 || echo "Failed to load nvidia/cuda/11.8"
module load cudnn 2>&1 || echo "Failed to load cudnn"
echo "Loaded modules:"
module list 2>&1
echo ""

echo "===== GPU Device Files ====="
echo "Checking /dev/nvidia* devices:"
ls -la /dev/nvidia* 2>/dev/null || echo "No /dev/nvidia* devices found"
echo ""
echo "Checking /proc/driver/nvidia/:"
ls -la /proc/driver/nvidia/ 2>/dev/null || echo "No /proc/driver/nvidia/ found"
echo ""

echo "===== GPU Hardware Detection ====="
echo "Running lspci | grep -i nvidia:"
lspci | grep -i nvidia || echo "No NVIDIA devices found with lspci"
echo ""
echo "Running lshw -C display (if available):"
lshw -C display 2>/dev/null || echo "lshw not available or no display devices"
echo ""

echo "===== NVIDIA Driver and Tools ====="
echo "Running nvidia-smi:"
nvidia-smi 2>&1 || echo "nvidia-smi failed or not available"
echo ""
echo "Running nvidia-ml-py test (if available):"
python3 -c "
try:
    import pynvml
    pynvml.nvmlInit()
    print(f'NVML initialized successfully')
    count = pynvml.nvmlDeviceGetCount()
    print(f'GPU count: {count}')
    for i in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        print(f'GPU {i}: {name}')
except Exception as e:
    print(f'NVML error: {e}')
" 2>&1 || echo "pynvml test failed"
echo ""

echo "===== CUDA Compiler ====="
echo "Running nvcc --version:"
nvcc --version 2>&1 || echo "nvcc not available"
echo ""
echo "Checking CUDA installation paths:"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
which nvcc 2>/dev/null || echo "nvcc not in PATH"
echo ""

echo "===== Environment Setup ====="
# Setup conda
echo "Setting up conda environment..."
eval "$(/dss/dsshome1/0A/di97jur/miniconda3/bin/conda shell.bash hook)"

echo "Python version:"
python --version
echo ""

echo "===== PyTorch CUDA Test ====="
echo "Testing PyTorch CUDA availability..."
python -c "
import sys
print('Python executable:', sys.executable)
print('Python version:', sys.version)
print()

try:
    import torch
    print('PyTorch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    print('CUDA version:', torch.version.cuda)
    print('cuDNN version:', torch.backends.cudnn.version())
    print('Number of GPUs:', torch.cuda.device_count())
    
    if torch.cuda.is_available():
        print('Current GPU:', torch.cuda.current_device())
        print('GPU name:', torch.cuda.get_device_name(0))
        print('GPU memory:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')
        
        # Test creating a tensor on GPU
        try:
            x = torch.randn(10, 10).cuda()
            print('Successfully created tensor on GPU')
            print('Tensor device:', x.device)
        except Exception as e:
            print('Failed to create tensor on GPU:', e)
    else:
        print('CUDA not available - running on CPU only')
        
except ImportError as e:
    print('PyTorch not installed:', e)
except Exception as e:
    print('PyTorch error:', e)
"
echo ""

echo "===== TensorFlow GPU Test ====="
echo "Testing TensorFlow GPU availability..."
python -c "
try:
    import tensorflow as tf
    print('TensorFlow version:', tf.__version__)
    print('GPU devices:', tf.config.list_physical_devices('GPU'))
    print('CUDA built with TensorFlow:', tf.test.is_built_with_cuda())
    print('GPU available:', tf.test.is_gpu_available())
except ImportError:
    print('TensorFlow not installed')
except Exception as e:
    print('TensorFlow error:', e)
"
echo ""

echo "===== GPU Memory Test ====="
echo "Checking GPU memory usage..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
else
    echo "nvidia-smi not available for memory check"
fi
echo ""

echo "===== Final Environment Check ====="
echo "Final CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Final LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

echo "===== GPU Test Completed ====="
echo "Test completed at: `date`"
