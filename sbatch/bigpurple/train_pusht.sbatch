#!/bin/bash
#SBATCH --partition=gl40s_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --job-name=train_pusht
#SBATCH --output=/gpfs/scratch/sd6701/personal/salt/dino_wm/logs/%x_%j.log
#SBATCH --error=/gpfs/scratch/sd6701/personal/salt/dino_wm/logs/%x_%j.err

# Trains a DINO-WM on PushT with a frozen DINOv2 ViT-S/14 encoder (conf/encoder/dino.yaml).
# Pretrained weights come from the torch.hub cache under TORCH_HOME.
#
# Test (50 rollouts, 1 epoch):
#   sbatch --partition=gl40s_dev --time=01:00:00 --export=ALL,N_ROLLOUT=50,EPOCHS=1 \
#     sbatch/bigpurple/train_pusht.sbatch
#
# Full training (all rollouts, 100 epochs):
#   sbatch sbatch/bigpurple/train_pusht.sbatch
#
# Resume from the last saved epoch. EPOCHS counts additional epochs, not the total:
#   sbatch --export=ALL,RUN_DIR=<run folder>,EPOCHS=<epochs left> \
#     sbatch/bigpurple/train_pusht.sbatch

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

ARGS=(
  ckpt_base_path=/gpfs/scratch/sd6701/personal/salt/dino_wm_runs
  env=pusht
  encoder=dino
  frameskip=5
  num_hist=3
  training.epochs="${EPOCHS:-100}"
  env.dataset.n_rollout="${N_ROLLOUT:-null}"
)
if [ -n "${RUN_DIR:-}" ]; then
  ARGS+=("hydra.run.dir=${RUN_DIR}")
fi

python train.py --config-name train.yaml "${ARGS[@]}"
