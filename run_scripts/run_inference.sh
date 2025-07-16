#!/bin/bash
#SBATCH --job-name=eval_retrieval_t5      # 作业名称
#SBATCH --output=logs/eval_t5_%j.out      # 标准输出日志文件
#SBATCH --error=logs/eval_t5_%j.err       # 标准错误日志文件
#SBATCH --ntasks=1                        # 启动的任务数
#SBATCH --cpus-per-task=4                 # 每个任务的 CPU 数
#SBATCH --gres=shard:16                   # 请求 1 块 GPU
#SBATCH --mem=32G                         # 请求 32GB 内存
#SBATCH --time=12:00:00                   # 最长运行时间

# 进入工作目录
cd /home/iiserver32/Workbench/fyb/adaptiveRAG/baselines/SmartRAG
start_time=$(date +%s)
echo "=== Starting evaluation at $(date) ==="

# 加载你的 conda 环境或模块（根据实际情况修改）
# conda activate smartrag   # 替换成你实际的 conda 环境名

echo "=== GPU Status Before Evaluation ==="
nvidia-smi

echo "=== CPU and Memory Usage ==="
free -h

echo "Running evaluate_retrieval_t5.py..."
echo "Step 1: Environment setup done. Starting script..."
python inference/evaluate_retrieval_t5.py \
    --base_model_path ckpt/flan-t5-large-warm-up-v2/checkpoint-13000 \
    --checkpoint /data2/fyb/rl4lm_exps/t5_large_three_ppo_debug_v4/checkpoints/checkpoint_19 \
    --dataset ambignq \
    --save_evaluate_path results/t5_large_three_ppo_debug_v4_test_output_ambignq | tee logs/eval_output_runtime.log

echo "Step 2: Script finished. Gathering final GPU usage..."
nvidia-smi
echo "Step 3: Evaluation complete."
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "=== Total runtime: ${duration} seconds ==="
echo "=== Evaluation completed at $(date) ==="hoa