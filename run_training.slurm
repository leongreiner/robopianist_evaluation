#!/bin/sh
#SBATCH --job-name=robopianist_train
#SBATCH --time=1-14:00:00
#SBATCH -o /dss/dsstbyfs02/pn52ru/pn52ru-dss-0000/di97jur/robopianist_evaluation/job_logs/train.%j.%N.out
#SBATCH -e /dss/dsstbyfs02/pn52ru/pn52ru-dss-0000/di97jur/robopianist_evaluation/job_logs/train.%j.%N.err
#SBATCH -D /dss/dsstbyfs02/pn52ru/pn52ru-dss-0000/di97jur/robopianist_evaluation/
#SBATCH --cluster=hpda2
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=hpda2_compute_gpu

echo "===== RoboPianist Training Job Info ===="
echo "Node List: " $SLURM_NODELIST
echo "jobID: " $SLURM_JOB_ID
echo "Partition: " $SLURM_JOB_PARTITION
echo "Submit directory:" $SLURM_SUBMIT_DIR
echo "Submit host:" $SLURM_SUBMIT_HOST
echo "In the directory: `pwd`"
echo "As the user: `whoami`"

# Create job logs directory
mkdir -p job_logs

echo "===== RoboPianist Training Job Info ====="
echo "Node List: " $SLURM_NODELIST
echo "jobID: " $SLURM_JOB_ID
echo "Partition: " $SLURM_JOB_PARTITION
echo "Submit directory:" $SLURM_SUBMIT_DIR
echo "Submit host:" $SLURM_SUBMIT_HOST
echo "In the directory: `pwd`"
echo "As the user: `whoami`"
echo "Date: $(date)"

# Setup conda FIRST to access the pianist environment
echo "===== Setting up Conda Environment ====="
eval "$(/dss/dsshome1/0A/di97jur/miniconda3/bin/conda shell.bash hook)"

# Activate the pianist environment (where PyTorch CUDA is available)
echo "===== Activating Conda Environment ====="
conda activate pianist
echo "Active environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

# NOW load CUDA module after conda environment is active
echo "===== Loading CUDA Modules ====="
module purge 2>/dev/null || true
module load cuda 2>/dev/null && echo "Successfully loaded CUDA module" || echo "Failed to load CUDA module"

echo "Currently loaded modules:"
module list 2>/dev/null || echo "Module list not available"

# Set CUDA environment variables AFTER loading module
echo "===== CUDA Environment Setup ====="
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Debug GPU information
echo "===== SLURM GPU Variables ====="
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "SLURM_GPU_BIND: $SLURM_GPU_BIND"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Check GPU device files
echo "===== GPU Device Files ====="
ls -la /dev/nvidia* 2>/dev/null || echo "No /dev/nvidia* devices found"

# Show GPU info
echo "===== nvidia-smi Test ====="
nvidia-smi || echo "nvidia-smi failed - checking if GPU drivers are available"

# Check if GPU is available via SLURM
if [ -n "$SLURM_GPUS_ON_NODE" ] && [ "$SLURM_GPUS_ON_NODE" -gt 0 ]; then
    echo "SLURM has allocated $SLURM_GPUS_ON_NODE GPU(s)"
    # Set CUDA_VISIBLE_DEVICES based on SLURM allocation
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        export CUDA_VISIBLE_DEVICES=0
        echo "Set CUDA_VISIBLE_DEVICES to 0"
    fi
else
    echo "WARNING: No GPUs allocated by SLURM, training will use CPU"
    export CUDA_VISIBLE_DEVICES=""
fi

# Verify final environment setup
echo "===== Final Environment Check ====="
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Test PyTorch CUDA availability before training
echo "===== Testing PyTorch CUDA availability ====="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
    print('✅ GPU available for training!')
else:
    print('❌ CUDA not available - PyTorch will use CPU only!')
    print('This means your PyTorch installation is CPU-only.')
    print('You need to install a CUDA-enabled PyTorch version.')
    print('Check: https://pytorch.org/get-started/locally/')
    import sys
    print(f'Python executable: {sys.executable}')
    try:
        import os
        print(f'LD_LIBRARY_PATH: {os.environ.get(\"LD_LIBRARY_PATH\", \"Not set\")}')
        print(f'CUDA_HOME: {os.environ.get(\"CUDA_HOME\", \"Not set\")}')
    except:
        pass
"

start_time=`date +%s`
echo "RoboPianist Training Started at "`date`

# Set additional environment variables
export TMPDIR=/tmp/di97jur

echo "Final CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Running parallel training scripts..."

# Run the training script
bash /dss/dsstbyfs02/pn52ru/pn52ru-dss-0000/di97jur/robopianist_evaluation/run_training.sh

echo "RoboPianist Training ended at "`date`
end_time=`date +%s`
total_time=$((end_time-start_time))
echo "Total training time: " $total_time " seconds ("$((total_time/3600))" hours)"
