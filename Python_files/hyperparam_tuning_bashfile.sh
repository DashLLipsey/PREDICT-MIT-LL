#!/bin/bash
#SBATCH --job-name=Conditional_encoder_hyperparam_tuning
#SBATCH --output=/home/dlipsey/MITLincolnLabs/logs/cond_encoder1234e1e2_hyperparam_tuning_%j.out
#SBATCH --time=03:00:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8


# Create logs directory if it doesn't exist
mkdir -p /home/dlipsey/MITLincolnLabs/logs

# Print some info
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run the hyperparameter tuning 
source /home/dlipsey/MITLincolnLabs/.venv/bin/activate 
cd /home/dlipsey/MITLincolnLabs/Python_files
python /home/dlipsey/MITLincolnLabs/Python_files/cond_enc_1234e1e2_hyperparam_tune_grid_search.py

echo "Finished at: $(date)"