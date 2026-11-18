#!/bin/bash
#SBATCH --partition=gpu4_dev
#SBATCH --gres=gpu:v100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --job-name=train_pusht_2gpu
#SBATCH --output=/gpfs/scratch/sd6701/personal/salt/dino_wm/logs/%x_%j.log
#SBATCH --error=/gpfs/scratch/sd6701/personal/salt/dino_wm/logs/%x_%j.err

# Trains a DINO-WM on PushT on 2 V100s with a frozen DINOv2 ViT-S/14 encoder.
# BATCH_SIZE is the total across GPUs, so 16 means 8 per GPU, which fit a single V100.
#
# Test (50 rollouts, 1 epoch):
#   sbatch --time=01:00:00 --export=ALL,N_ROLLOUT=50,EPOCHS=1,WANDB_MODE=online \
#     sbatch/bigpurple/train_pusht_2gpu.sbatch
#
# ENCODER picks conf/encoder/<name>.yaml: dino (default), vjepa_student, or vit_random
# (frozen ViT-S/14, random weights, seeded by init_seed so plan.py rebuilds the same encoder).
# 3 L40S, 8 per GPU:
#   sbatch --partition=gl40s_short --gres=gpu:3 --time=12:00:00 --job-name=train_pusht_random \
#     --export=ALL,ENCODER=vit_random,NUM_GPUS=3,BATCH_SIZE=24 sbatch/bigpurple/train_pusht_2gpu.sbatch

set -euo pipefail

# When several copies are queued under one job name, the first to start cancels the rest.
# Only this user's pending jobs with exactly this name are affected.
scancel --user="$USER" --name="$SLURM_JOB_NAME" --state=PENDING || true

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

CKPT_BASE=/gpfs/scratch/sd6701/personal/salt/dino_wm_runs
# Both processes must share one folder: Hydra's default name uses the time each one starts.
RUN_DIR="${RUN_DIR:-${CKPT_BASE}/outputs/$(date +%Y-%m-%d)/job_${SLURM_JOB_ID}}"

# Each process starts its own data workers, so 8 per process uses the 16 CPUs.
python -m accelerate.commands.launch \
  --multi_gpu \
  --num_processes "${NUM_GPUS:-2}" \
  --num_machines 1 \
  --mixed_precision no \
  --dynamo_backend no \
  --main_process_port $((20000 + SLURM_JOB_ID % 20000)) \
  train.py --config-name train.yaml \
  ckpt_base_path="${CKPT_BASE}" \
  env=pusht \
  encoder="${ENCODER:-dino}" \
  frameskip=5 \
  num_hist=3 \
  training.batch_size="${BATCH_SIZE:-16}" \
  training.epochs="${EPOCHS:-100}" \
  env.dataset.n_rollout="${N_ROLLOUT:-null}" \
  env.num_workers=8 \
  "hydra.run.dir=${RUN_DIR}"
