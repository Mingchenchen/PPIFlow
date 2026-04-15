# Pipeline Steps

This document describes every step class under `steps/` and how it is used by `pipeline.py`.

## Execution Context

The pipeline imports and orchestrates these step classes:

- `PPIFlowStep`
- `MPNNStep`
- `AbMPNNStep`
- `FlowpackerStep`
- `AF3scoreStep`
- `FilterStep`
- `RosettaFixStep`
- `PartialStep`
- `ReFoldStep`
- `DockQStep`
- `RosettaRelaxStep`
- `RankStep`
- `ReportStep`

## 1. `PPIFlowStep`

Purpose:

- Generates initial backbone structures with PPIFlow
- Supports `binder`, `antibody`, and `nanobody` modes

Back-end scripts:

- `tools/PPIFlow/sample_binder.py`
- `tools/PPIFlow/sample_antibody_nanobody.py`

Main inputs:

- `gentype`
- `output_dir`
- `model_weights`
- `config`
- `name`
- Binder mode:
  - `input_pdb`
  - `target_chain`
  - `binder_chain`
  - `specified_hotspots`
  - `sample_hotspot_rate_min`
  - `sample_hotspot_rate_max`
  - `samples_min_length`
  - `samples_max_length`
  - `samples_per_target`
- Antibody/nanobody mode:
  - `antigen_pdb`
  - `antigen_chain`
  - `framework_pdb`
  - `heavy_chain`
  - `light_chain` for antibody
  - `cdr_length`
  - `samples_per_target`

Used in pipeline:

- Stage 1 only

Typical output:

- `stage1/ppiflow_output`

## 2. `MPNNStep`

Purpose:

- Runs ProteinMPNN sequence design for binder workflows

Supported protocols:

- `base`: standard design
- `bias`: design with amino-acid bias
- `fix`: design with fixed positions

Back-end scripts:

- `tools/ProteinMPNN/helper_scripts/parse_multiple_chains.py`
- `tools/ProteinMPNN/helper_scripts/assign_fixed_chains.py`
- `tools/ProteinMPNN/helper_scripts/make_bias_AA.py`
- `tools/ProteinMPNN/helper_scripts/make_fixed_positions_dict.py`
- `tools/ProteinMPNN/protein_mpnn_run.py`

Main inputs:

- `protocol`
- `folder_with_pdbs`
- `output_dir`
- `num_seq_per_target`
- `sampling_temp`
- `batch_size`
- `chains_to_design`
- `omit_AAs`
- `use_soluble_model`
- `bias_AA_list` and `bias_list` for `bias`
- `fixed_positions` for `fix`

Used in pipeline:

- `MPNNStep_stage1` for binder stage 1
- `MPNNStep_stage2` for binder stage 2

Important behavior:

- Stage 2 automatically switches between `base` and `fix` depending on whether fixed positions are found for a given partial sample.

Typical outputs:

- `stage1/mpnn_output`
- `stage2/mpnn_output`
- FASTA files later collected into `mpnn_pdbs/mpnn_seqs.csv`

## 3. `AbMPNNStep`

Purpose:

- Runs AbMPNN sequence design for antibody and nanobody workflows

Back-end scripts:

- `tools/AbMPNN/make_fix_csv.py`
- `tools/AbMPNN/protein_mpnn_run.py`

Main inputs:

- `folder_with_pdbs`
- `output_dir`
- `checkpoint_dir`
- `model_name`
- `chain_list`
- `num_seq_per_target`
- `sampling_temp`
- `seed`
- `batch_size`
- `omit_AAs`
- `fix_positions`

Used in pipeline:

- `AbMPNNStep_stage1`
- `AbMPNNStep_stage2`

Important behavior:

- Builds a `fixed_positions.csv`
- Can merge extra fixed positions passed from stage 2

Typical outputs:

- `stage1/abmpnn_output`
- `stage2/abmpnn_output`

## 4. `FlowpackerStep`

Purpose:

- Packs side chains for designed sequences based on `mpnn_seqs.csv`

Back-end script:

- `tools/flowpacker/sampler_pdb_antibody.py`

Main inputs:

- `config_path`
- `output_dir`
- `csv_path`
- `pdb_dir`
- `python_executable`
- `use_gt_masks`

Used in pipeline:

- `FlowpackerStep_stage1`
- `FlowpackerStep_stage2`

Expected upstream inputs:

- `pdb_dir`: symlinked PDB directory assembled by the scheduler
- `csv_path`: generated `mpnn_pdbs/mpnn_seqs.csv`

Typical output:

- `flowpacker_output/run_1`

## 5. `AF3scoreStep`

Purpose:

- Evaluates structures with the AF3Score workflow
- Produces AF3-style metrics used for downstream filtering

Main inputs:

- `input_pdb_dir`
- `output_dir`
- `db_dir`
- `model_dir`
- `num_workers`

Typical internal workflow:

1. Prepare single-chain CIF files and JSON inputs
2. Convert PDB inputs to JAX/H5 representations
3. Run AF3Score inference
4. Collect metrics into CSV

Used in pipeline:

- `AF3scoreStep_stage1`
- `AF3scoreStep_stage2`

Important output:

- `af3score_metrics.csv`

## 6. `FilterStep`

Purpose:

- Filters structures according to numeric metric thresholds in a CSV file
- Creates symlinks for passing candidates into the next stage

Main inputs:

- `pdb_dir`
- `csv_file`
- `outdir`
- `filters`
- `output_csv`
- `filename_col`

Filter syntax:

- Exact numeric value:
  - `iptm: 0.7`
- Single rule string:
  - `iptm: "> 0.7"`
- Multiple rule string:
  - `pae: "<= 10, > 1"`


Used in pipeline:

- `FilterStep_stage1`
- `FilterStep_stage2`

Important pipeline detail:

- The scheduler passes `filename_col="description"` because AF3Score outputs use that column name for structure IDs.

Typical outputs:

- `stage1/filtered_iptm07`
- `stage2/filtered_iptm08`

## 7. `RosettaFixStep`

Purpose:

- Runs Rosetta-based fix/relax preparation on filtered structures
- Extracts interface energy information that is later converted to fixed positions

Supported modes:

- `binder`
- `nanobody`
- `antibody`

Main inputs:

- `gentype`
- `output_dir`
- `rosetta_bin`
- `native_xml`
- `interface_dist`
- `plot_toggle`
- `overwrite`

Used in pipeline:

- Stage 2, before `PartialStep`

Important outputs:

- Per-PDB Rosetta work directories
- Interface energy files used to generate `stage2/fixed_positions.csv`

## 8. `PartialStep`

Purpose:

- Performs partial redesign guided by fixed positions generated from Rosetta analysis

Supported modes:

- `binder`
- `nanobody`
- `antibody`

Back-end scripts:

- `tools/PPIFlow/sample_binder_partial.py`
- `tools/PPIFlow/sample_antibody_nanobody_partial.py`

Main inputs:

- `gentype`
- `pdb_dir`
- `fixed_positions_csv`
- `output_dir`
- `model_weights`
- `config`
- `samples_per_target`
- `start_t`

Binder-specific parameters:

- `target_chain`
- `binder_chain`
- `sample_hotspot_rate_min`
- `sample_hotspot_rate_max`

Antibody/nanobody-specific parameters:

- `antigen_chain`
- `heavy_chain`
- `light_chain`
- `retry_Limit`

Used in pipeline:

- Stage 2 only

Important behavior:

- Loads per-PDB fixed positions from CSV
- Extracts hotspot-like residues from B-factors when needed

Typical output:

- `stage2/partial_output`

## 9. `ReFoldStep`

Purpose:

- Refolds filtered stage 2 structures using an AlphaFold3-based workflow

Main inputs:

- `pdb_dir`
- `output_dir`
- `af3_python`
- `db_dir`
- `model_dir`
- `hmmer_path`
- `run_MSA`
- `remove_template`
- `run_AF3_inference`
- `seed_num`
- `cuda_bin`
- `cuda_lib`
- `xla_client_mem_fraction`

Workflow summary:

1. Convert PDBs to FASTA
2. Generate AF3 JSON jobs
3. Run MSA/data preparation
4. Optionally strip templates
5. Run AF3 inference
6. Export metrics

Used in pipeline:

- Stage 2 after `FilterStep_stage2`

Important output:

- `af3_iptm@1.csv` used later by `RankStep`

## 10. `DockQStep`

Purpose:

- Compares refolded models against the filtered reference complexes using DockQ

Main inputs:

- `reference_dir`
- `model_dir`
- `output_csv`
- `dockq_exe`
- `dockq_args`
- `recursive`

Used in pipeline:

- Stage 2 after `ReFoldStep`

Important output:

- `dockq_results.csv`

Pipeline note:

- The scheduler first reformats refold outputs into a DockQ-friendly model directory structure.

## 11. `RosettaRelaxStep`

Purpose:

- Runs Rosetta complex relaxation on filtered stage 2 structures

Back-end script:

- `steps/rosetta/relax_complex.py`

Main inputs:

- `pdb_dir`
- `output_dir`
- `batch_idx`
- `dump_pdb`

Used in pipeline:

- Stage 2 after `FilterStep_stage2`

Important output:

- `rosetta_complex_0.csv` or another batch-indexed CSV, consumed by `RankStep`

## 12. `RankStep`

Purpose:

- Merges DockQ, Rosetta, and AF3 results
- Keeps the best model per target above a DockQ threshold
- Computes final ranking fields
- Copies selected PDBs into a final design output directory

Main inputs:

- `gentype`
- `dockq_csv`
- `rosetta_csv`
- `af3_csv`
- `filtered_pdb_dir`
- `output_dir`
- `dockq_threshold`
- `output_csv_name`

Used in pipeline:

- Late Stage 2

Important outputs:

- `design_output/<output_csv_name>`
- `design_output/pdbs/`

Ranking note:

- Binder mode ranks using `iptm`
- Antibody/nanobody mode looks for `iptm_A_C` or `chain_A_iptm`

## 13. `ReportStep`

Purpose:

- Generates an HTML report for the final design set

Main inputs:

- `gentype`
- `name`
- `output_dir`
- `stage1_dir`
- `stage2_dir`
- `af3score_stage1_csv`
- `af3score_stage2_csv`
- `af3_refold_csv`
- `ranked_csv`
- `output_csv_name`
- `report_filename`

Used in pipeline:

- Final Stage 2 step

Report contents:

- Stage counts
- Metric histograms
- Top-ranked PDB viewer
- Ranked design table

Typical output:

- `design_output/design_report.html`

## Typical Step Chains

### Binder

`PPIFlowStep -> MPNNStep -> FlowpackerStep -> AF3scoreStep -> FilterStep -> RosettaFixStep -> PartialStep -> MPNNStep -> FlowpackerStep -> AF3scoreStep -> FilterStep -> ReFoldStep -> DockQStep -> RosettaRelaxStep -> RankStep -> ReportStep`

### Antibody / Nanobody

`PPIFlowStep -> AbMPNNStep -> FlowpackerStep -> AF3scoreStep -> FilterStep -> RosettaFixStep -> PartialStep -> AbMPNNStep -> FlowpackerStep -> AF3scoreStep -> FilterStep -> ReFoldStep -> DockQStep -> RosettaRelaxStep -> RankStep -> ReportStep`


