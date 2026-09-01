# FlowPacker

This repository contains the codebase for the codebase for FlowPacker, our method for protein side-chain packing with 
flow matching - [Link to bioRxiv.](https://www.biorxiv.org/content/10.1101/2024.07.05.602280v1)

## Installation

Install the environment using pip:

`pip install -r requirements.txt`

Note that you should install the appropriate `torch` and `torch_geometric` packages based on your CUDA version.

## Inference

Currently, we only support batch inference on a folder containing the PDBs of interest.

All PDB files found in the folder specified under `test_path` in `./config/inference/base.yaml` is used for inference.

To perform single-file inference, you can simply place the PDB file in a folder in isolation and run the command below.

To run inference:

`python sampler_pdb.py base <name>`

where `<name>` is just any name to label the sampling run.

By default, the model will run with masks based on missing coordinates found in the supplied PDBs for benchmarking purposes.
In case you want to generate samples with all chi angles regardless of whether they are missing in the PDB file, add the flag `--use_gt_masks True`.

For conditioning on specific regions, you have to add the flag `--inpaint` which is just a list of residues that you wish to design (all other residues will be fixed to the native PDB file).
For instance, if you want to design res 10-25 of chain A, 40 of chain B, and the entire chain C, the format is as follows:

`--inpaint A_10-25/B_40/C`

Note that we use standard 1-residue indexing (first residue is 1), and the residue ranges are inclusive from start to end.

NOTE: we do some preprocessing of the structures that may cause minor differences from the input structure. The inputs
should not contain any of the following:
1. unknown or nonstandard amino acids - these will simply just be ignored
2. overlapping Ca atoms - all of these will be dropped
3. Extremely long proteins/complexes (tested on RTX3060 12GB VRAM up to 1000 residues) - GPU memory issues

We provide two checkpoints: one trained on BC40 (`bc40.pth`) and one trained on a PDB snapshot clustered at 40% sequence identity, denoted as PDB-S40 in the paper (`cluster.pth`).
We recommend using `cluster.pth` for most cases.

Checkpoints with other ablations can be provided upon request.

To turn on and off confidence sampling, simply add (or remove) the path to the confidence model checkpoint in the config file.

Samples can be found in `./samples/<name>/`, with `best_run` containing the PDBs selected by the confidence model.

After sampling, `output_dict.pkl` contains all metrics for the inference run that are also described in the paper.
(Note: metrics are calculated based on available input side-chain conformations - missing sidechains are dropped)

`--save_traj` can be added to save the entire sampling trajectory rather than just the final sample.

## Training

To train the flow model, run the command

`python trainer.py vf`

In the config file, you can specify clusters in `cluster_path` by using the `*_cluster.tsv` output from MMSeqs2 - the dataloader will sample
one random member from each cluster every epoch. Simply keep it empty to parse all structures every epoch (each structure becomes its own cluster).

To train the confidence model, first sample any number of samples from the trained flow model (we use 8).

Then, edit `sample_path` in `./config/training/confidence.yaml` to the respective path, and run

`python trainer_conf.py confidence`

The folder structure for confidence samples should be something like below:

```
<sample_path>
│
└───run_1
│   │   PDB1.pdb
│   │   PDB2.pdb
|   |   ...
...
|
└───run_8
│   │   PDB1.pdb
│   │   PDB2.pdb
|   |   ...
```

DDP training can be turned on for training (currently configured for SLURM) by using the flag `--ddp True`

The training/test datasets are all publicly available datasets (details in paper), but can be made available upon request.
We provide the list of PDBs and corresponding chains used for training with the BC40 and PDB-S40 datasets under `./data`.

Please contact me at jinsub.lee@mail.utoronto.ca for any questions or concerns.