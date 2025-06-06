#!/bin/bash

# Activate conda environment first
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate llm-uncertainty

# Clear any existing CUDA device settings
unset CUDA_VISIBLE_DEVICES

# Set up paths for CUDA 12.6
CONDA_PREFIX="$HOME/miniconda/envs/llm-uncertainty"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"

# Set up library paths in the correct order
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:\
$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:\
$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:\
$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:\
$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:\
/usr/lib/x86_64-linux-gnu:\
$LD_LIBRARY_PATH"

# Make device 0 visible
export CUDA_VISIBLE_DEVICES=0

# Print current settings
echo "=== CUDA Environment ==="
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CONDA_PREFIX: $CONDA_PREFIX"

# Run CUDA check
python cuda_init.py 