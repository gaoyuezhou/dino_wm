#!/bin/bash
#SBATCH --partition=gl40s_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=plan_pusht
#SBATCH --output=/gpfs/scratch/sd6701/personal/salt/dino_wm/logs/%x_%j.log
#SBATCH --error=/gpfs/scratch/sd6701/personal/salt/dino_wm/logs/%x_%j.err

# Evaluates a PushT world model with CEM inside MPC (conf/plan_pusht.yaml), 1xL40S.
# plan.py loads ${CKPT_BASE}/outputs/${MODEL_NAME}/{hydra.yaml,checkpoints/model_${MODEL_EPOCH}.pth}.
# The checkpoint has no encoder, so plan.py rebuilds it from hydra.yaml: dinov2_vits14 from the
# torch.hub cache, or the frozen salt student from its checkpoint_path (conf/encoder/vjepa_student.yaml).
# sbatch needs the logs folder to exist before the job starts: mkdir -p logs
#
# Default: the shared pretrained checkpoint (dino_wm_checkpoints/outputs/pusht).
#
# One of your own training runs, e.g. dino_wm_runs/outputs/2026-11-02/job_12345:
#   sbatch --export=ALL,CKPT_BASE=/gpfs/scratch/sd6701/personal/salt/dino_wm_runs,MODEL_NAME=2026-11-02/job_12345 \
#     sbatch/bigpurple/plan_pusht.sbatch
#
# Test (5 tasks):
#   sbatch --partition=gl40s_dev --time=00:30:00 --export=ALL,N_EVALS=5,CKPT_BASE=...,MODEL_NAME=... \
#     sbatch/bigpurple/plan_pusht.sbatch
#
# Full evaluation (50 tasks) is the default. MODEL_EPOCH picks a checkpoint other than model_latest.pth.
# Results: plan_outputs/<timestamp>_<model_name>_gH5/logs.json (success_rate, mean_state_dist) and W&B.

set -euo pipefail

cd /gpfs/scratch/sd6701/personal/salt/dino_wm

eval "$(conda shell.bash hook)"
conda activate /gpfs/scratch/sd6701/conda-envs/dino_wm

export PYTHONNOUSERSITE=1
# Home is over its file quota: keep caches and W&B files on scratch.
export XDG_CACHE_HOME=/gpfs/scratch/sd6701/.cache
export TORCH_HOME=/gpfs/scratch/sd6701/.cache/torch
export XDG_CONFIG_HOME=/gpfs/scratch/sd6701/.config
export WANDB_DIR=/gpfs/scratch/sd6701/wandb
export WANDB_CONFIG_DIR=/gpfs/scratch/sd6701/.config/wandb
export WANDB_CACHE_DIR=/gpfs/scratch/sd6701/.cache/wandb
mkdir -p "$WANDB_DIR" "$WANDB_CONFIG_DIR" "$WANDB_CACHE_DIR"
export DATASET_DIR=/gpfs/scratch/sd6701/personal/salt/datasets/dinowm_dataset
export HYDRA_FULL_ERROR=1

CKPT_BASE="${CKPT_BASE:-/gpfs/scratch/sd6701/personal/salt/dino_wm_checkpoints}"
MODEL_NAME="${MODEL_NAME:-pusht}"
MODEL_EPOCH="${MODEL_EPOCH:-latest}"
RUN_DIR="${CKPT_BASE}/outputs/${MODEL_NAME}"
if [ ! -f "${RUN_DIR}/hydra.yaml" ] || [ ! -f "${RUN_DIR}/checkpoints/model_${MODEL_EPOCH}.pth" ]; then
  echo "No hydra.yaml or checkpoints/model_${MODEL_EPOCH}.pth under ${RUN_DIR}" >&2
  exit 1
fi

python plan.py --config-name plan_pusht.yaml \
  ckpt_base_path="${CKPT_BASE}" \
  model_name="${MODEL_NAME}" \
  model_epoch="${MODEL_EPOCH}" \
  n_evals="${N_EVALS:-50}"
