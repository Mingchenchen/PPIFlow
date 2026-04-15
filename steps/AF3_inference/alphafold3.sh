#!/bin/bash
#SBATCH -p gpu43,gpu41
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -o logs/%x-%j.out
#SBATCH -J af3_inf
#SBATCH -x c06b14n05

# 1. 正确激活 conda 环境
# conda activate /lustre/grp/cmclab/share/wangd/env/alphafold3

# 2. 设置环境变量
export PATH=/lustre/grp/cmclab/share/wangd/hmmer/bin:$PATH
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_CLIENT_MEM_FRACTION=0.95

start_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Job started at $start_time"

input_dir=$1
output_dir=$2
iptm_threshold=$3
min_samples_above_threshold=$4

script_dir=$(dirname "$(realpath "$0")")

/lustre/grp/cmclab/share/wangd/env/alphafold3/bin/python "$script_dir/run_alphafold_dy.py" \
  --db_dir=/lustre/grp/cmclab/share/wangd/af3_data \
  --model_dir=/lustre/grp/cmclab/share/chenmc/Alphafold3params \
  --iptm_threshold=$iptm_threshold \
  --min_samples_above_threshold=$min_samples_above_threshold \
  --input_dir=$input_dir \
  --output_dir=$output_dir \
  --run_data_pipeline=False \
  --run_inference=true

end_time=$(date "+%Y-%m-%d %H:%M:%S")
echo "Job ended at $end_time"
start_sec=$(date -d "$start_time" +%s)
end_sec=$(date -d "$end_time" +%s)
duration=$((end_sec - start_sec))
echo "Job duration: $duration seconds"
