#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 <input_pdb_dir> <csv_file> <output_dir> <num_of_jobs>"
    echo ""
    echo "Arguments:"
    echo "  input_pdb_dir   Directory containing input PDB files"
    echo "  csv_file        Path to CSV file (columns: link_name,seq,seq_idx)"
    echo "  output_dir      Directory to store the results"
    echo "  num_of_jobs     Number of jobs to split/run"
    echo ""
    exit 1
}

# Check if correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    usage
fi

REPO='demo_scripts/flowpacker_af3score'

echo "========== Job started at: $(date) =========="
start_time=$(date +%s)

input_pdb_dir=$1        # e.g. input
csv_file=$2             # link_name,seq,seq_idx
output_dir=$3
num_of_jobs=$4           

# !!!!!!!!!!!!!!!!! Run Flowpacker !!!!!!!!!!!!!!!!!!!!!!!!!!!

# Create directories required for flowpacker
output_flowpacker_base="$output_dir/flowpacker"
mkdir -p "$output_flowpacker_base"

flowpacker_input_pdb_batch="$output_flowpacker_base/input_pdb_batch"
output_dir_yml="$output_flowpacker_base/batch_yml"
output_dir_flowpacker="$output_flowpacker_base/flowpacker_outputs"
flowpacker_log="$output_flowpacker_base/logs"

# Create output directories
mkdir -p "$flowpacker_input_pdb_batch" "$output_dir_yml" "$output_dir_flowpacker" "$flowpacker_log"

# Step 1: Split batch
/lustre/grp/cmclab/share/anaconda3/envs/jaxaf2/bin/python $REPO/1-split_batch.py \
  --base_path "$input_pdb_dir" \
  --output_folder_base "$flowpacker_input_pdb_batch" \
  --num_of_jobs "$num_of_jobs"

# Step 2: Write batch folder to batch yml
/lustre/grp/cmclab/share/anaconda3/envs/jaxaf2/bin/python $REPO/2-run_flowpacker.py \
  --base_pdb_path "$flowpacker_input_pdb_batch" \
  --output_yaml_dir "$output_dir_yml" \

# Step 3: Submit flowpacker jobs

declare -a job_ids=()
for subfolder in "$flowpacker_input_pdb_batch"/*; do
  if [ -d "$subfolder" ]; then
    folder_name=$(basename "$subfolder")
    echo "Processing subdirectory: $folder_name"

    submit_output_flowpacker=$(sbatch --output="$flowpacker_log/${folder_name}-%j.out" \
    $REPO/3-submit_flowpacker_gpu_jobs.sh  \
      "$output_dir_yml/${folder_name}.yml" \
      "$output_dir_flowpacker" \
      "$csv_file"
      )
   

    if [[ "$submit_output_flowpacker" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        job_id="${BASH_REMATCH[1]}"
        echo "Submitted JobID=$job_id"
        job_ids+=("$job_id")
    else
        echo "Submission failed: $submit_output_flowpacker"
    fi
    
    sleep 0.05
  fi
done

# Step 4: Wait for all tasks to complete
echo "Waiting for all flowpacker tasks to complete..."

while true; do
  unfinished_jobs=0
  for job_id in "${job_ids[@]}"; do
    job_state=$(squeue -j "$job_id" -h -o "%T")
    if [[ "$job_state" == "RUNNING" || "$job_state" == "PENDING" ]]; then
      unfinished_jobs=$((unfinished_jobs + 1))
    fi
  done

  if [ "$unfinished_jobs" -eq 0 ]; then
    echo "All flowpacker tasks completed"
    break
  else
    echo "Remaining flowpacker tasks: $unfinished_jobs..."
    sleep 30
  fi

done

echo "All flowpacker tasks completed, starting af3score"


# !!!!!!!!!!!!!!!!! Run prepare_af3score !!!!!!!!!!!!!!!!!!!!!!!!!!!

# Create directories required for af3score
output_af3score_base="$output_dir/af3score_base_outputs"
mkdir -p "$output_af3score_base"

af3_input_batch="$output_af3score_base/af3_input_batch"
output_dir_cif="$output_af3score_base/single_chain_cif"
save_csv="$output_af3score_base/single_seq.csv"
output_dir_json="$output_af3score_base/json"
output_dir_jax="$af3_input_batch/jax"
output_dir_af3score="$output_af3score_base/af3score_outputs"
metric_csv="$output_af3score_base/af3score_metrics.csv"
jax_log_dir="$output_af3score_base/logs/jax"
af3score_log_dir="$output_af3score_base/logs/af3score"


# Create output directories
mkdir -p "$af3_input_batch" "$output_dir_cif" "$output_dir_jax"  "$output_dir_json" "$output_dir_af3score" "$jax_log_dir" "$af3score_log_dir"

# Step 1: get seq, json and split batch by length
echo "Step 1: get seq, json and split batch by length"
/lustre/grp/cmclab/share/anaconda3/envs/jaxaf2/bin/python $REPO/4.1-prepare_get_json.py \
      --input_dir "$output_dir_flowpacker/run_1" \
      --output_dir_cif "$output_dir_cif" \
      --save_csv "$save_csv" \
      --output_dir_json "$output_dir_json" \
      --batch_dir "$af3_input_batch" \
      --num_jobs "$num_of_jobs"

# =========== Step 2: Submit prepare_jax jobs ===========
echo "Step 2: Submit prepare_jax jobs"
declare -a job_ids_prepare=()
for subfolder in "$af3_input_batch"/pdb/*; do
  if [ -d "$subfolder" ]; then
    folder_name=$(basename "$subfolder")
    echo "Preparing: $folder_name"

    submit_prepare_output=$(sbatch --output="$jax_log_dir/${folder_name}-%j.out" \
    $REPO/4.2-submit_prepare_jax.sh \
      "$subfolder" \
      "$output_dir_jax/$folder_name" 
)

    if [[ "$submit_prepare_output" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
      job_id="${BASH_REMATCH[1]}"
      echo "  JobID=$job_id"
      job_ids_prepare+=("$job_id")
    else
      echo "Submission failed: $submit_prepare_output"
    fi

    sleep 0.05
  fi
done




# =========== Step 3: Wait for all prepare_af3score tasks to complete ===========
echo "Step 3: Waiting for all prepare_jax to complete..."
while true; do
  unfinished_jobs=0
  for job_id in "${job_ids_prepare[@]}"; do
    job_state=$(squeue -j "$job_id" -h -o "%T")
    if [[ "$job_state" == "RUNNING" || "$job_state" == "PENDING" ]]; then
      unfinished_jobs=$((unfinished_jobs + 1))
    fi
  done

   if [ "$unfinished_jobs" -eq 0 ]; then
    echo "All prepare_jax tasks completed"
    break
  else
    echo "Remaining prepare_jax tasks: $unfinished_jobs..."
    sleep 30
  fi
done


# =========== Check quantity of generated h5 files ===========
echo "Checking h5 file generation status..."

# Iterate through all batch folders
for subfolder in "$af3_input_batch"/pdb/*; do
  if [ -d "$subfolder" ]; then
    folder_name=$(basename "$subfolder")
    pdb_dir="$subfolder"
    h5_dir="$output_dir_jax/$folder_name"

    # Get pdb and h5 filenames (remove extensions)
    pdb_names=($(find "$pdb_dir" -name "*.pdb" -exec basename {} .pdb \; | sort))
    h5_names=($(find "$h5_dir" -name "*.h5" -exec basename {} .h5 \; | sort))

    # Convert to string set for comparison
    missing=()
    for name in "${pdb_names[@]}"; do
      if ! printf "%s\n" "${h5_names[@]}" | grep -qx "$name"; then
        missing+=("$name")
      fi
    done

    if [ "${#missing[@]}" -eq 0 ]; then
      echo "$folder_name: All h5 files generated (${#pdb_names[@]} files)"
    else
      echo "$folder_name: Missing ${#missing[@]} h5 files:"
      for miss in "${missing[@]}"; do
        echo "    - $miss"
      done
    fi
  fi
done



 # =========== Step 4: Submit af3score GPU inference jobs ===========
echo "Step 4: Submit af3score inference jobs"
declare -a job_ids_af3score=()
for subfolder in "$af3_input_batch"/json/*; do
  if [ -d "$subfolder" ]; then
    folder_name=$(basename "$subfolder")
    echo "Inference: $folder_name"

    submit_output=$(sbatch --output="$af3score_log_dir/${folder_name}-%j.out" \
    $REPO/5-submit_af3score.sh \
      "$af3_input_batch/json/$folder_name" \
      "$af3_input_batch/jax/$folder_name" \
      "$output_dir_af3score" 
      )

    if [[ "$submit_output" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
      job_id="${BASH_REMATCH[1]}"
      echo "  JobID=$job_id"
      job_ids_af3score+=("$job_id")
    else
      echo "Submission failed: $submit_output"
    fi

    sleep 0.05
  fi
done

 # =========== Step 5: Wait for all af3score tasks to complete ===========
echo "Step 5: Waiting for all af3score inference tasks to complete..."
while true; do
  unfinished_jobs=0
  for job_id in "${job_ids_af3score[@]}"; do
    job_state=$(squeue -j "$job_id" -h -o "%T")
    if [[ "$job_state" == "RUNNING" || "$job_state" == "PENDING" ]]; then
      unfinished_jobs=$((unfinished_jobs + 1))
    fi
  done

  if [ "$unfinished_jobs" -eq 0 ]; then
    echo "All af3score GPU inference tasks completed"
    break
  else
    echo "Remaining af3score tasks: $unfinished_jobs..."
    sleep 60
  fi
done

echo "All processes completed! Output directory: $output_af3score_base"

# Step 6: Extract all metrics
echo "Step 6: Start extracting metrics..."
python $REPO/easy_get_metrics.py \
 "$output_dir_af3score" \
 "$metric_csv"

# Step 7: Check if csv is complete
expected_count=$(ls "$output_flowpacker_base/after_pdbs"/*pdb 2>/dev/null | wc -l)
# expected_flowpacker_count=$(ls "$save_flowpacker_output_dir"/*pdb 2>/dev/null | wc -l)

echo "$metric_csv"
actual_count=$(cat "$metric_csv" | wc -l)
actual_count=$((actual_count - 1))  # Subtract header row

columns_non_empty=$(python3 <<EOF
import pandas as pd
df = pd.read_csv('$metric_csv')
print(all(df.notna().all()))
EOF
)


if [[ "$actual_count" -eq "$expected_count"  && "$columns_non_empty" == "True" ]]; then
  echo "Check passed: Record count matches and no empty columns, deleting intermediate folders"
  # rm -rf "$output_dir_cif"   "$flowpacker_input_pdb_batch" "$output_dir_json" "$af3_input_batch" "$output_dir_yml" "$flowpacker_input_pdb_batch" "$output_flowpacker_base/after_pdbs_batch"
else
  echo "Check failed: Intermediate folders will not be deleted"
  echo "Expected count: $expected_count, Actual count: $actual_count, All columns non-empty: $columns_non_empty"
fi

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "========== Job finished at: $(date) =========="
echo "========== Total runtime: ${elapsed} seconds =========="