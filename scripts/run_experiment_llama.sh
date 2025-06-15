#!/bin/bash

#SBATCH --job-name=experiment-llm-uncertainty-llama
#SBATCH --partition=gpu-v100
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-courses-dsait4095

# Load modules:
module load 2023r1
module load cuda/11.6
module load openmpi
module load miniconda3
module load ffmpeg

echo "Modules loaded, loading conda environment..."

# Activate the conda environment
conda activate IST-ASR

echo "Conda environment activated, running script..."

cd llm-uncertainty
# Run your script
python src/subjecticity_classification.py --model_name meta-llama --samples_limit 100 --binary_classification

# Deactivate the environment when done
conda deactivate