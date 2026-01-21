#!/bin/bash
#SBATCH -p gpu41
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -J af3score

echo "========== Job started at: $(date) =========="
start_time=$(date +%s)

export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_CLIENT_MEM_FRACTION=0.95

batch_json_dir=$1
batch_h5_dir=$2
output_dir=$3
buckets=$(basename "$batch_json_dir" | grep -oE '[0-9]+$')


echo "Running on: $batch_json_dir  $batch_h5_dir  $buckets -> $output_dir" 

/lustre/grp/cmclab/share/wangd/env/alphafold3/bin/python demo_scripts/flowpacker_af3score/run_af3score.py \
  --db_dir=/lustre/grp/cmclab/share/wangd/af3_data \
  --model_dir=/lustre/grp/cmclab/share/chenmc/Alphafold3params \
  --batch_json_dir="$batch_json_dir" \
  --batch_h5_dir="$batch_h5_dir" \
  --output_dir="$output_dir" \
  --run_data_pipeline=False \
  --run_inference=true \
  --init_guess=true \
  --num_samples=1 \
  --buckets="$buckets" \
  --write_cif_model=False \
  --write_summary_confidences=true \
  --write_full_confidences=true \
  --write_best_model_root=false \
  --write_ranking_scores_csv=false \
  --write_terms_of_use_file=false \
  --write_fold_input_json_file=false


end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "========== Job finished at: $(date) =========="
echo "========== Total runtime: ${elapsed} seconds =========="