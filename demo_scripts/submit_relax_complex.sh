#!/bin/bash
#SBATCH -p cpu2,cpu1,cpu_short
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -o logs/relax/complex_score-%j.out
#SBATCH -J logs/relax/complex_score
source /lustre/grp/cmclab/qinxy/new_miniconda3/etc/profile.d/conda.sh
conda activate rosetta

PDB_DIR=$1
OUTPUT_DIR=$2
BATCH_IDX=$3

echo "Running relax on $PDB_DIR → $OUTPUT_DIR (batch $BATCH_IDX)"

python demo_scripts/relax_complex.py \
  --pdb_dir ${PDB_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --batch_idx ${BATCH_IDX} \
  --dump_pdb True \
  --relax True \
  --fixbb False \
  --max_iter 170
