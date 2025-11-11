#!/bin/bash

#SBATCH --job-name=tml-benchmark
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --partition=hpg-b200
#SBATCH --gpus=2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sivarama.swamyna@ufl.edu
#SBATCH --time=16:00:00
#SBATCH --output=logs/%j_tml_log.log

module purge
module load cuda/12.2
ml conda
source ../venv/bin/activate

echo "=== Python & CUDA Environment ==="
python --version
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
echo ""
nvidia-smi
echo ""
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Starting benchmark..."
python3 benchmark.py
