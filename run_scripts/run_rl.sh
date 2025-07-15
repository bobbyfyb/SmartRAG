#!/bin/bash

#===================== Slurm 资源配置 =====================#
#SBATCH --job-name=rl-flan-t5-large
#SBATCH --output=logs/rl-flan-t5-large-%j.out
#SBATCH --error=logs/rl-flan-t5-large-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=5-00:00:00
#SBATCH --gres=shard:16  # flan-t5-large建议占满1张A100（16个shard）

#===================== 环境设置 =====================#
echo "Running on $(hostname) with Job ID $SLURM_JOB_ID"
# source activate smartrag_env  # 请替换为你的 conda 环境名称
# export HF_HOME=/data2/fyb/huggingface_cache
# export WANDB_PROJECT="sft-flan-t5-large"

#===================== 加载训练脚本 =====================#
# 使用 accelerate 启动，确保 `accelerate config` 已在当前环境配置好
python rl/scripts/training/train_text_generation.py \
--config_path rl/scripts/training/task_configs/three/t5_ppo_debug.yml \
--experiment_name t5_large_three_ppo_debug_v4 \
--base_path_to_store_results /data2/fyb \
--log_to_wandb

echo "Training finished with exit code $?"