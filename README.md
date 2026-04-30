# PPIFlow Pipeline

![](./model.png)

This repository provides a two-stage protein design pipeline built around **PPIFlow**. It supports three design modes:

- `binder`
- `antibody`
- `nanobody`

The pipeline orchestrates structure generation, sequence design, side-chain packing, AF3score scoring, filtering, partial redesign, AF3 refolding, ranking, and report generation through `pipeline.py`.

For a given design task, the pipeline generates a directory named `design_output`, which contains the PDB files of the designed structures, sequences, computed evaluation metrics for each design, and a design report. See [Design Demo]([/example/design_output](/example/design_output)).

See [tool/PPIFlow/README.md](/tool/PPIFlow/README.md) for PPIFlow only.

## Environment Installation

```bash
git clone https://github.com/Mingchenchen/PPIFlow.git
cd PPIFlow

bash Install.sh

# Install Rosetta
wget https://downloads.rosettacommons.org/downloads/academic/2024/wk09/rosetta.binary.linux.release-371.tar.bz2
tar -xvf rosetta.binary.linux.release-371.tar.bz2
```

Download the PPIFlow checkpoints from [Google Drive](https://drive.google.com/drive/folders/1BcIBUL2yq1gOchHfN68-AcZK3hiMAMVN?usp=drive_link)

| Task Type | Checkpoint Path |
|------------|-----------------|
| Binder | `binder.ckpt` |
| Antibody | `antibody.ckpt` |
| Nanobody | `nanobody.ckpt` |
| Monomer or Motif Scaffolding  | `monomer.ckpt` |

## Pipeline Overview

`pipeline.py` reads two YAML files:

- `task.yaml`: task-level metadata, input structures, generation mode, output location, and step enable/disable switches
- `steps.yaml`: executable paths and per-step parameters

The scheduler always creates:

- `output_base_dir/stage1`
- `output_base_dir/stage2`

and can run either stage independently or both stages in sequence.

### Stage 1

Typical Stage 1 flow:

1. `PPIFlowStep`
2. `MPNNStep_stage1` or `AbMPNNStep_stage1`
3. `FlowpackerStep_stage1`
4. `AF3scoreStep_stage1`
5. `FilterStep_stage1`

### Stage 2

Typical Stage 2 flow:

1. `RosettaFixStep`
2. `PartialStep`
3. `MPNNStep_stage2` or `AbMPNNStep_stage2`
4. `FlowpackerStep_stage2`
5. `AF3scoreStep_stage2`
6. `FilterStep_stage2`
7. `ReFoldStep`
8. `DockQStep`
9. `RosettaRelaxStep`
10. `RankStep`
11. `ReportStep`

The exact order and output directories are implemented directly in `pipeline.py`.

## Run `pipeline.py`



### Full run

```bash
python pipeline.py --task example/task_binder.yaml --steps example/steps_config/steps_binder.yaml
```

Before running the pipeline, you need to download the Flowpacker model weights([Flowpacker](https://gitlab.com/mjslee0921/flowpacker)), AF3 model weights and database([AlphaFold3](https://github.com/google-deepmind/alphafold3)). Many paths in the example `steps.yaml` files must be replaced with your local paths.

> [!NOTE]
> - The pipeline runs on a single GPU.  
>   For large-scale inference, it is recommended to perform **batch processing across multiple GPUs**, with each GPU handling approximately **1000–2000 designs**.
>
> - The following steps（Stage 2） are **CPU-intensive and time-consuming**:
>   - `ReFoldStep (MSA)`
>   - `DockQStep`
>   - `RosettaRelaxStep`
>
>   To accelerate computation, it is recommended to run these steps separately on **high-performance CPUs**.


### Stage 1 only

```bash
python pipeline.py \
  --task example/task_binder.yaml \
  --steps example/steps_config/steps_binder.yaml \
  --stage 1
```

### Stage 2 only

```bash
python pipeline.py \
  --task example/task_binder.yaml \
  --steps example/steps_config/steps_binder.yaml \
  --stage 2
```

CLI arguments:

- `--task`: path to task configuration YAML
- `--steps`: path to step configuration YAML
- `--stage`: optional; `1` or `2`

If `--stage` is omitted, the pipeline runs both stages.

## `task.yaml` Configuration

`task.yaml` contains two top-level sections:

- `task`
- `steps`

`task` stores run metadata and biological inputs. `steps` is a map of booleans that enables or disables each pipeline step inside the scheduler.

### Minimal Structure

```yaml
task:
  name: "example"
  gentype: "binder"
  output_base_dir: "./outputs/example"

steps:
  PPIFlowStep: true
  MPNNStep_stage1: true
  FlowpackerStep_stage1: true
  AF3scoreStep_stage1: true
  FilterStep_stage1: true
  RosettaFixStep: false
  PartialStep: false
  MPNNStep_stage2: false
  FlowpackerStep_stage2: false
  AF3scoreStep_stage2: false
  FilterStep_stage2: false
  ReFoldStep: false
  DockQStep: false
  RosettaRelaxStep: false
  RankStep: false
  ReportStep: false
```

### Common `task` Fields

- `name`: run name used as the design prefix
- `gentype`: one of `binder`, `antibody`, `nanobody`
- `output_base_dir`: root output directory for this run
- `samples_per_target`: number of structures sampled by PPIFlow
- `specified_hotspots`: optional comma-separated hotspot residues such as `B67,B78,B99`

### Binder-Specific Fields

These are consumed by `PPIFlowStep` when `gentype: binder`:

- `input_pdb`: input complex PDB
- `target_chain`: target protein chain ID
- `binder_chain`: binder chain ID
- `sample_hotspot_rate_min`
- `sample_hotspot_rate_max`
- `samples_min_length`
- `samples_max_length`

Example:

```yaml
task:
  name: "CD3d"
  gentype: "binder"
  input_pdb: "example/target_and_framework_pdb/CD3d.pdb"
  target_chain: "B"
  binder_chain: "A"
  specified_hotspots: "B67,B78,B99"
  samples_min_length: 50
  samples_max_length: 60
  samples_per_target: 10000
  output_base_dir: "../test_pipeline_binder"
```

### Antibody-Specific Fields

These are used when `gentype: antibody`:

- `antigen_pdb`
- `antigen_chain`
- `framework_pdb`
- `heavy_chain`
- `light_chain`
- `cdr_length`

Example:

```yaml
task:
  name: "IL13"
  gentype: "antibody"
  antigen_pdb: "example/target_and_framework_pdb/1IJZ_IL13.pdb"
  antigen_chain: "C"
  specified_hotspots: "C10,C13,C14,C100,C106,C107"
  framework_pdb: "example/target_and_framework_pdb/6nou_scfv_framework.pdb"
  heavy_chain: "A"
  light_chain: "B"
  cdr_length: "CDRH1,8-8,CDRH2,8-8,CDRH3,10-20,CDRL1,6-9,CDRL2,3-3,CDRL3,9-11"
  samples_per_target: 10
  output_base_dir: "../test_pipeline_antibody"
```

### Nanobody-Specific Fields

These are used when `gentype: nanobody`:

- `antigen_pdb`
- `antigen_chain`
- `framework_pdb`
- `heavy_chain`
- `cdr_length`

Example:

```yaml
task:
  name: "IL13"
  gentype: "nanobody"
  antigen_pdb: "example/target_and_framework_pdb/1IJZ_IL13.pdb"
  antigen_chain: "C"
  specified_hotspots: "C10,C13,C14,C100,C106,C107"
  framework_pdb: "example/target_and_framework_pdb/7eow_nanobody_framework.pdb"
  heavy_chain: "A"
  cdr_length: "CDRH1,8-8,CDRH2,8-8,CDRH3,9-21"
  samples_per_target: 10
  output_base_dir: "../test_pipeline_nanobody"
```


## `steps.yaml` Configuration

`steps.yaml` stores per-step runtime parameters and external tool paths. The scheduler loads one config block per step name and passes it into the corresponding step class.

### Recommended Pattern

Start from one of:

- `example/steps_config/steps_binder.yaml`
- `example/steps_config/steps_antibody.yaml`
- `example/steps_config/steps_nanobody.yaml`

These files show the expected keys for each design mode.

See [steps/README.md](/steps/README.md) for a step-by-step description of every pipeline component.


## Output Directories

The scheduler uses fixed directory names under `output_base_dir`. The main layout is documented in the file header of `pipeline.py`, including:

- `stage1/ppiflow_output`
- `stage1/mpnn_pdbs`
- `stage1/mpnn_output` or `stage1/abmpnn_output`
- `stage1/flowpacker_output`
- `stage1/af3score_output`
- `stage1/filtered_iptm07`
- `stage2/rosetta_fix_output`
- `stage2/fixed_positions.csv`
- `stage2/before_partial_pdbs`
- `stage2/partial_output`
- `stage2/mpnn_pdbs`
- `stage2/mpnn_output` or `stage2/abmpnn_output`
- `stage2/flowpacker_output`
- `stage2/af3score_output`
- `stage2/filtered_iptm08`
- `stage2/refold_output`
- `stage2/dockq_output`
- `stage2/rosetta_relax_output`
- `design_output`

## Notes

- Stage 2 expects Stage 1 outputs unless you manually prepare the required intermediate files.
- Many paths in the example `steps.yaml` files must be replaced with your local paths.
- The repository examples are the best starting templates for new runs.

## Cite
```
@article {yu2026ppiflow,
	author = {Yu, Qilin and Guo, Liangyue and Qin, Xiayan and Huang, Xikun and Tian, Baihui and Wang, Hongzhun and Liu, Yu and Lang, Yunzhi and Wang, Di and Shen, Zhouhanyu and Lin, Jie and Chen, Mingchen},
	title = {High-Affinity Protein Binder Design via Flow Matching and In Silico Maturation},
	year = {2026},
	doi = {10.64898/2026.01.19.700484},
	journal = {bioRxiv}
}
```
