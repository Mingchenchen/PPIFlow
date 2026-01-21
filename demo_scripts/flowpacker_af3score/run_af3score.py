# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""AlphaFold 3 structure prediction script.

AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

To request access to the AlphaFold 3 model parameters, follow the process set
out at https://github.com/google-deepmind/alphafold3. You may only use these
if received directly from Google. Use is subject to terms of use available at
https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
"""

from collections.abc import Callable, Iterable, Sequence
import csv
import dataclasses
import functools
import multiprocessing
import os
import pathlib
import shutil
import string
import textwrap
import time
import typing
from typing import Protocol, Self, TypeVar, overload
import gc
import jax

from absl import app
from absl import flags
from alphafold3.common import base_config
from alphafold3.common import folding_input
from alphafold3.common import resources
from alphafold3.constants import chemical_components
import alphafold3.cpp
from alphafold3.data import featurisation
from alphafold3.data import pipeline
from alphafold3.jax.attention import attention
from alphafold3.model import features
from alphafold3.model import params
from alphafold3.model import post_processing
from alphafold3.model.components import base_model
from alphafold3.model.components import utils
from alphafold3.model.diffusion import model as diffusion_model
import haiku as hk
from jax import numpy as jnp
import numpy as np
from model_manager_correct import get_optimized_runner, OptimizedModelRunner




_HOME_DIR = pathlib.Path(os.environ.get('HOME'))
_DEFAULT_MODEL_DIR = _HOME_DIR / 'models'
_DEFAULT_DB_DIR = _HOME_DIR / 'public_databases'


# Input and output paths.
_JSON_PATH = flags.DEFINE_string(
    'json_path',
    None,
    'Path to the input JSON file.',
)
_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    None,
    'Path to the directory containing input JSON files.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'Path to a directory where the results will be saved.',
)

# Add batch processing arguments
_BATCH_JSON_DIR = flags.DEFINE_string(
    'batch_json_dir',
    None,
    'Path to directory containing multiple JSON files for batch processing.',
)
_BATCH_H5_DIR = flags.DEFINE_string(
    'batch_h5_dir',
    None,
    'Path to directory containing multiple H5 files for batch processing.',
)

MODEL_DIR = flags.DEFINE_string(
    'model_dir',
    _DEFAULT_MODEL_DIR.as_posix(),
    'Path to the model to use for inference.',
)

_FLASH_ATTENTION_IMPLEMENTATION = flags.DEFINE_enum(
    'flash_attention_implementation',
    default='triton',
    enum_values=['triton', 'cudnn', 'xla'],
    help=(
        "Flash attention implementation to use. 'triton' and 'cudnn' uses a"
        ' Triton and cuDNN flash attention implementation, respectively. The'
        ' Triton kernel is fastest and has been tested more thoroughly. The'
        " Triton and cuDNN kernels require Ampere GPUs or later. 'xla' uses an"
        ' XLA attention implementation (no flash attention) and is portable'
        ' across GPU devices.'
    ),
)

# Control which stages to run.
_RUN_DATA_PIPELINE = flags.DEFINE_bool(
    'run_data_pipeline',
    True,
    'Whether to run the data pipeline on the fold inputs.',
)
_RUN_INFERENCE = flags.DEFINE_bool(
    'run_inference',
    True,
    'Whether to run inference on the fold inputs.',
)

# Binary paths.
_JACKHMMER_BINARY_PATH = flags.DEFINE_string(
    'jackhmmer_binary_path',
    shutil.which('jackhmmer'),
    'Path to the Jackhmmer binary.',
)
_NHMMER_BINARY_PATH = flags.DEFINE_string(
    'nhmmer_binary_path',
    shutil.which('nhmmer'),
    'Path to the Nhmmer binary.',
)
_HMMALIGN_BINARY_PATH = flags.DEFINE_string(
    'hmmalign_binary_path',
    shutil.which('hmmalign'),
    'Path to the Hmmalign binary.',
)
_HMMSEARCH_BINARY_PATH = flags.DEFINE_string(
    'hmmsearch_binary_path',
    shutil.which('hmmsearch'),
    'Path to the Hmmsearch binary.',
)
_HMMBUILD_BINARY_PATH = flags.DEFINE_string(
    'hmmbuild_binary_path',
    shutil.which('hmmbuild'),
    'Path to the Hmmbuild binary.',
)

# Database paths.
DB_DIR = flags.DEFINE_multi_string(
    'db_dir',
    (_DEFAULT_DB_DIR.as_posix(),),
    'Path to the directory containing the databases. Can be specified multiple'
    ' times to search multiple directories in order.',
)

_SMALL_BFD_DATABASE_PATH = flags.DEFINE_string(
    'small_bfd_database_path',
    '${DB_DIR}/bfd-first_non_consensus_sequences.fasta',
    'Small BFD database path, used for protein MSA search.',
)
_MGNIFY_DATABASE_PATH = flags.DEFINE_string(
    'mgnify_database_path',
    '${DB_DIR}/mgy_clusters_2022_05.fa',
    'Mgnify database path, used for protein MSA search.',
)
_UNIPROT_CLUSTER_ANNOT_DATABASE_PATH = flags.DEFINE_string(
    'uniprot_cluster_annot_database_path',
    '${DB_DIR}/uniprot_all_2021_04.fa',
    'UniProt database path, used for protein paired MSA search.',
)
_UNIREF90_DATABASE_PATH = flags.DEFINE_string(
    'uniref90_database_path',
    '${DB_DIR}/uniref90_2022_05.fa',
    'UniRef90 database path, used for MSA search. The MSA obtained by '
    'searching it is used to construct the profile for template search.',
)
_NTRNA_DATABASE_PATH = flags.DEFINE_string(
    'ntrna_database_path',
    '${DB_DIR}/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta',
    'NT-RNA database path, used for RNA MSA search.',
)
_RFAM_DATABASE_PATH = flags.DEFINE_string(
    'rfam_database_path',
    '${DB_DIR}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta',
    'Rfam database path, used for RNA MSA search.',
)
_RNA_CENTRAL_DATABASE_PATH = flags.DEFINE_string(
    'rna_central_database_path',
    '${DB_DIR}/rnacentral_active_seq_id_90_cov_80_linclust.fasta',
    'RNAcentral database path, used for RNA MSA search.',
)
_PDB_DATABASE_PATH = flags.DEFINE_string(
    'pdb_database_path',
    '${DB_DIR}/mmcif_files',
    'PDB database directory with mmCIF files path, used for template search.',
)
_SEQRES_DATABASE_PATH = flags.DEFINE_string(
    'seqres_database_path',
    '${DB_DIR}/pdb_seqres_2022_09_28.fasta',
    'PDB sequence database path, used for template search.',
)

# Number of CPUs to use for MSA tools.
_JACKHMMER_N_CPU = flags.DEFINE_integer(
    'jackhmmer_n_cpu',
    min(multiprocessing.cpu_count(), 8),
    'Number of CPUs to use for Jackhmmer. Default to min(cpu_count, 8). Going'
    ' beyond 8 CPUs provides very little additional speedup.',
)
_NHMMER_N_CPU = flags.DEFINE_integer(
    'nhmmer_n_cpu',
    min(multiprocessing.cpu_count(), 8),
    'Number of CPUs to use for Nhmmer. Default to min(cpu_count, 8). Going'
    ' beyond 8 CPUs provides very little additional speedup.',
)

# Compilation buckets.
_BUCKETS = flags.DEFINE_list(
    'buckets',
    # pyformat: disable
    ['256', '512', '768', '1024', '1280', '1536', '2048', '2560', '3072',
     '3584', '4096', '4608', '5120'],
    # pyformat: enable
    'Strictly increasing order of token sizes for which to cache compilations.'
    ' For any input with more tokens than the largest bucket size, a new bucket'
    ' is created for exactly that number of tokens.',
)

# Add command line arguments
_INIT_GUESS = flags.DEFINE_bool(
    'init_guess',
    True,
    'Whether to use initial guess in the diffusion model.',
)

_INIT_PATH = flags.DEFINE_string(
    'path',
    '',
    'Path to the initial structure file.',
)

_NUM_SAMPLES = flags.DEFINE_integer(
    'num_samples',
    5,
    'Number of samples to generate for each prediction.',
)

# Output file control parameters
_WRITE_CIF_MODEL = flags.DEFINE_bool(
    'write_cif_model',
    True,
    'Whether to write model.cif files in sample directories.',
)

_WRITE_SUMMARY_CONFIDENCES = flags.DEFINE_bool(
    'write_summary_confidences', 
    True,
    'Whether to write summary_confidences.json files in sample directories.',
)

_WRITE_FULL_CONFIDENCES = flags.DEFINE_bool(
    'write_full_confidences',
    True, 
    'Whether to write confidences.json files in sample directories.',
)

# Control other optional outputs
_WRITE_BEST_MODEL_ROOT = flags.DEFINE_bool(
    'write_best_model_root',
    False,
    'Whether to write best model files to root output directory.',
)

_WRITE_RANKING_SCORES_CSV = flags.DEFINE_bool(
    'write_ranking_scores_csv',
    False,
    'Whether to write ranking_scores.csv file.',
)

_WRITE_TERMS_OF_USE_FILE = flags.DEFINE_bool(
    'write_terms_of_use_file',
    False,
    'Whether to write TERMS_OF_USE.md files.',
)

_WRITE_FOLD_INPUT_JSON_FILE = flags.DEFINE_bool(
    'write_fold_input_json_file',
    False,
    'Whether to write fold input JSON file.',
)


class ConfigurableModel(Protocol):
  """A model with a nested config class."""

  class Config(base_config.BaseConfig):
    ...

  def __call__(self, config: Config) -> Self:
    ...

  @classmethod
  def get_inference_result(
      cls: Self,
      batch: features.BatchDict,
      result: base_model.ModelResult,
      target_name: str = '',
  ) -> Iterable[base_model.InferenceResult]:
    ...


ModelT = TypeVar('ModelT', bound=ConfigurableModel)


def make_model_config(
    *,
    model_class: type[ModelT] = diffusion_model.Diffuser,
    flash_attention_implementation: attention.Implementation = 'triton',
):
  config = model_class.Config()
  if hasattr(config, 'global_config'):
    config.global_config.flash_attention_implementation = (
        flash_attention_implementation
    )
  return config


class ModelRunner(OptimizedModelRunner):
    """Use optimized ModelRunner"""
    
    def __init__(self, model_class, config, device, model_dir):
        # Use global singleton pattern
        global _global_runner
        if _global_runner is None:
            _global_runner = OptimizedModelRunner(model_class, config, device, model_dir)
        
        # Copy attributes from global instance
        self.__dict__.update(_global_runner.__dict__)

# Add global variable
_global_runner = None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
  """Stores the inference results (diffusion samples) for a single seed.

  Attributes:
    seed: The seed used to generate the samples.
    inference_results: The inference results, one per sample.
    full_fold_input: The fold input that must also include the results of
      running the data pipeline - MSA and templates.
  """

  seed: int
  inference_results: Sequence[base_model.InferenceResult]
  full_fold_input: folding_input.Input


# Define a global CCD variable at the module level
global_ccd = None

def predict_structure(
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    buckets: Sequence[int] | None = None,
    init_guess: bool = True,
    path: str = '',
    num_samples: int = 5,
    global_ccd = None,  # Add global CCD parameter
) -> Sequence[ResultsForSeed]:
  """Runs the full inference pipeline to predict structures for each seed."""

  featurisation_start_time = time.time()
  
  # Use global CCD if provided, otherwise create a new one
  ccd = global_ccd
  if ccd is None:
    ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
    
  featurised_examples = featurisation.featurise_input(
      fold_input=fold_input, buckets=buckets, ccd=ccd, verbose=True
  )
  
  all_inference_results = []
  for seed, example in zip(fold_input.rng_seeds, featurised_examples):
    rng_key = jax.random.PRNGKey(seed)
    result = model_runner.run_inference(
        example, 
        rng_key, 
        init_guess=init_guess, 
        path=path,
        num_samples=num_samples 
    )
    inference_results = model_runner.extract_structures(
        batch=example, result=result, target_name=fold_input.name
    )
    all_inference_results.append(
        ResultsForSeed(
            seed=seed,
            inference_results=inference_results,
            full_fold_input=fold_input,
        )
    )
  return all_inference_results


def write_fold_input_json(
    fold_input: folding_input.Input,
    output_dir: os.PathLike[str] | str,
) -> None:
  """Writes the input JSON to the output directory."""
  os.makedirs(output_dir, exist_ok=True)
  with open(
      os.path.join(output_dir, f'{fold_input.sanitised_name()}_data.json'), 'wt'
  ) as f:
    f.write(fold_input.to_json())


def write_selective_output(
    inference_result: base_model.InferenceResult,
    output_dir: os.PathLike[str] | str,
) -> None:
  """Selectively write inference result to directory."""
  processed_result = post_processing.post_process_inference_result(inference_result)
  
  if _WRITE_CIF_MODEL.value:
    with open(os.path.join(output_dir, 'model.cif'), 'wb') as f:
      f.write(processed_result.cif)
  
  if _WRITE_SUMMARY_CONFIDENCES.value:
    with open(os.path.join(output_dir, 'summary_confidences.json'), 'wb') as f:
      f.write(processed_result.structure_confidence_summary_json)
  
  if _WRITE_FULL_CONFIDENCES.value:
    with open(os.path.join(output_dir, 'confidences.json'), 'wb') as f:
      f.write(processed_result.structure_full_data_json)


def write_outputs(
    all_inference_results: Sequence[ResultsForSeed],
    output_dir: os.PathLike[str] | str,
    job_name: str,
) -> None:
  """Writes outputs to the specified output directory."""
  ranking_scores = []
  max_ranking_score = None
  max_ranking_result = None

  os.makedirs(output_dir, exist_ok=True)
  
  for results_for_seed in all_inference_results:
    seed = results_for_seed.seed
    for sample_idx, result in enumerate(results_for_seed.inference_results):
      # Only create sample directory and write selected files
      sample_dir = os.path.join(output_dir, f'seed-{seed}_sample-{sample_idx}')
      os.makedirs(sample_dir, exist_ok=True)
      
      # Use custom selective output function
      write_selective_output(
          inference_result=result, 
          output_dir=sample_dir
      )
      
      ranking_score = float(result.metadata['ranking_score'])
      ranking_scores.append((seed, sample_idx, ranking_score))
      if max_ranking_score is None or ranking_score > max_ranking_score:
        max_ranking_score = ranking_score
        max_ranking_result = result

  # Optional: write best model to root directory
  if max_ranking_result is not None and _WRITE_BEST_MODEL_ROOT.value:
    if _WRITE_TERMS_OF_USE_FILE.value:
      output_terms = (
          pathlib.Path(alphafold3.cpp.__file__).parent / 'OUTPUT_TERMS_OF_USE.md'
      ).read_text()
    else:
      output_terms = None
      
    post_processing.write_output(
        inference_result=max_ranking_result,
        output_dir=output_dir,
        terms_of_use=output_terms,
        name=job_name,
    )
    
  # Optional: write ranking scores CSV
  if _WRITE_RANKING_SCORES_CSV.value and ranking_scores:
    with open(os.path.join(output_dir, 'ranking_scores.csv'), 'wt') as f:
      writer = csv.writer(f)
      writer.writerow(['seed', 'sample', 'ranking_score'])
      writer.writerows(ranking_scores)


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
) -> folding_input.Input:
  ...


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
) -> Sequence[ResultsForSeed]:
  ...


def replace_db_dir(path_with_db_dir: str, db_dirs: Sequence[str]) -> str:
  """Replaces the DB_DIR placeholder in a path with the given DB_DIR."""
  template = string.Template(path_with_db_dir)
  if 'DB_DIR' in template.get_identifiers():
    for db_dir in db_dirs:
      path = template.substitute(DB_DIR=db_dir)
      if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f'{path_with_db_dir} with ${{DB_DIR}} not found in any of {db_dirs}.'
    )
  if not os.path.exists(path_with_db_dir):
    raise FileNotFoundError(f'{path_with_db_dir} does not exist.')
  return path_with_db_dir


def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner | None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    init_guess: bool = True,
    path: str = '',
    num_samples: int = 5,
    global_ccd = None,  # Add global CCD parameter
) -> folding_input.Input | Sequence[ResultsForSeed]:
  """Runs data pipeline and/or inference on a single fold input.

  Args:
    fold_input: Fold input to process.
    data_pipeline_config: Data pipeline config to use. If None, skip the data
      pipeline.
    model_runner: Model runner to use. If None, skip inference.
    output_dir: Output directory to write to.
    buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
      of the model. If None, calculate the appropriate bucket size from the
      number of tokens. If not None, must be a sequence of at least one integer,
      in strictly increasing order. Will raise an error if the number of tokens
      is more than the largest bucket size.

  Returns:
    The processed fold input, or the inference results for each seed.

  Raises:
    ValueError: If the fold input has no chains.
  """
  print(f'Processing fold input {fold_input.name}')

  if not fold_input.chains:
    raise ValueError('Fold input has no chains.')

  if model_runner is not None:
    # If we're running inference, check we can load the model parameters before
    # (possibly) launching the data pipeline.
    print('Checking we can load the model parameters...')
    _ = model_runner.model_params

  if data_pipeline_config is None:
    print('Skipping data pipeline...')
  else:
    print('Running data pipeline...')
    fold_input = pipeline.DataPipeline(data_pipeline_config).process(fold_input)

  print(f'Output directory: {output_dir}')
  if _WRITE_FOLD_INPUT_JSON_FILE.value:
    print(f'Writing model input JSON to {output_dir}')
    write_fold_input_json(fold_input, output_dir)
  else:
    print('Skipping writing fold input JSON')

  if model_runner is None:
    print('Skipping inference...')
    output = fold_input
  else:
    print(
        f'Predicting 3D structure for {fold_input.name} for seed(s)'
        f' {fold_input.rng_seeds}...'
    )
    all_inference_results = predict_structure(
        fold_input=fold_input,
        model_runner=model_runner,
        buckets=buckets,
        init_guess=init_guess,
        path=path,
        num_samples=num_samples,
        global_ccd=global_ccd  # Pass global CCD
    )
    print(
        f'Writing outputs for {fold_input.name} for seed(s)'
        f' {fold_input.rng_seeds}...'
    )
    write_outputs(
        all_inference_results=all_inference_results,
        output_dir=output_dir,
        job_name=fold_input.sanitised_name(),
    )
    output = all_inference_results

  print(f'Done processing fold input {fold_input.name}.')
  return output


def main(_):
  # Check for batch processing mode
  batch_mode = (_BATCH_JSON_DIR.value is not None and _BATCH_H5_DIR.value is not None)
  single_mode = (_JSON_PATH.value is not None or _INPUT_DIR.value is not None)
  
  if batch_mode and single_mode:
    raise ValueError(
        'Cannot use batch mode (--batch_json_dir and --batch_h5_dir) and single mode '
        '(--json_path or --input_dir) at the same time.'
    )
  
  if not batch_mode and not single_mode:
    raise ValueError(
        'Must specify either batch mode (--batch_json_dir and --batch_h5_dir) or '
        'single mode (--json_path or --input_dir).'
    )

  if not _RUN_INFERENCE.value and not _RUN_DATA_PIPELINE.value:
    raise ValueError(
        'At least one of --run_inference or --run_data_pipeline must be'
        ' set to true.'
    )

  # Process in single mode
  if single_mode:
    if _JSON_PATH.value is None == _INPUT_DIR.value is None:
      raise ValueError(
          'Exactly one of --json_path or --input_dir must be specified in single mode.'
      )

    if _INPUT_DIR.value is not None:
      fold_inputs = folding_input.load_fold_inputs_from_dir(
          pathlib.Path(_INPUT_DIR.value)
      )
    elif _JSON_PATH.value is not None:
      fold_inputs = folding_input.load_fold_inputs_from_path(
          pathlib.Path(_JSON_PATH.value)
      )
  # Process in batch mode
  else:
    # Check if directories exist
    if not os.path.isdir(_BATCH_JSON_DIR.value):
      raise ValueError(f'Batch JSON directory does not exist: {_BATCH_JSON_DIR.value}')
    if not os.path.isdir(_BATCH_H5_DIR.value):
      raise ValueError(f'Batch H5 directory does not exist: {_BATCH_H5_DIR.value}')
    
    # Get all JSON files from the batch directory
    batch_json_files = [f for f in os.listdir(_BATCH_JSON_DIR.value) if f.endswith('.json')]
    if not batch_json_files:
      raise ValueError(f'No JSON files found in {_BATCH_JSON_DIR.value}')
    
    print(f'Found {len(batch_json_files)} JSON files for batch processing')
    
    # We'll process each JSON file individually later

  # Make sure we can create the output directory before running anything.
  try:
    os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
  except OSError as e:
    print(f'Failed to create output directory {_OUTPUT_DIR.value}: {e}')
    raise

  if _RUN_INFERENCE.value:
    # Fail early on incompatible devices, but only if we're running inference.
    gpu_devices = jax.local_devices(backend='gpu')
    if gpu_devices and float(gpu_devices[0].compute_capability) < 8.0:
      raise ValueError(
          'There are currently known unresolved numerical issues with using'
          ' devices with compute capability less than 8.0. See '
          ' https://github.com/google-deepmind/alphafold3/issues/59 for'
          ' tracking.'
      )

  notice = textwrap.wrap(
      'Running AlphaFold 3. Please note that standard AlphaFold 3 model'
      ' parameters are only available under terms of use provided at'
      ' https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md.'
      ' If you do not agree to these terms and are using AlphaFold 3 derived'
      ' model parameters, cancel execution of AlphaFold 3 inference with'
      ' CTRL-C, and do not use the model parameters.',
      break_long_words=False,
      break_on_hyphens=False,
      width=80,
  )
  print('\n'.join(notice))
  if _RUN_DATA_PIPELINE.value:
    expand_path = lambda x: replace_db_dir(x, DB_DIR.value)
    data_pipeline_config = pipeline.DataPipelineConfig(
        jackhmmer_binary_path=_JACKHMMER_BINARY_PATH.value,
        nhmmer_binary_path=_NHMMER_BINARY_PATH.value,
        hmmalign_binary_path=_HMMALIGN_BINARY_PATH.value,
        hmmsearch_binary_path=_HMMSEARCH_BINARY_PATH.value,
        hmmbuild_binary_path=_HMMBUILD_BINARY_PATH.value,
        small_bfd_database_path=expand_path(_SMALL_BFD_DATABASE_PATH.value),
        mgnify_database_path=expand_path(_MGNIFY_DATABASE_PATH.value),
        uniprot_cluster_annot_database_path=expand_path(
            _UNIPROT_CLUSTER_ANNOT_DATABASE_PATH.value
        ),
        uniref90_database_path=expand_path(_UNIREF90_DATABASE_PATH.value),
        ntrna_database_path=expand_path(_NTRNA_DATABASE_PATH.value),
        rfam_database_path=expand_path(_RFAM_DATABASE_PATH.value),
        rna_central_database_path=expand_path(_RNA_CENTRAL_DATABASE_PATH.value),
        pdb_database_path=expand_path(_PDB_DATABASE_PATH.value),
        seqres_database_path=expand_path(_SEQRES_DATABASE_PATH.value),
        jackhmmer_n_cpu=_JACKHMMER_N_CPU.value,
        nhmmer_n_cpu=_NHMMER_N_CPU.value,
    )
  else:
    print('Skipping running the data pipeline.')
    data_pipeline_config = None

  if _RUN_INFERENCE.value:
    devices = jax.local_devices(backend='gpu')
    print(f'Found local devices: {devices}')

    # Key modification: create ModelRunner only once, and it uses global model manager
    print('Initializing global model manager...')
    global_init_start = time.time()
    
    model_runner = ModelRunner(
        model_class=diffusion_model.Diffuser,
        config=make_model_config(
            flash_attention_implementation=typing.cast(
                attention.Implementation, _FLASH_ATTENTION_IMPLEMENTATION.value
            )
        ),
        device=devices[0],
        model_dir=pathlib.Path(MODEL_DIR.value),
    )
    
    global_init_time = time.time() - global_init_start
    print(f'Global model initialization completed, time taken: {global_init_time:.2f} seconds')
    
  else:
    print('Skipping running model inference.')
    model_runner = None

  if batch_mode:
    # Load global CCD only once
    print("Loading chemical component dictionary (CCD) once for all inputs...")
    global_ccd = chemical_components.cached_ccd(user_ccd=None)
    
    # Batch processing: now each inference uses the same compiled model
    print(f'Processing {len(batch_json_files)} files in batch mode')
    batch_start_time = time.time()
    
    for i, json_file in enumerate(batch_json_files):
        file_start_time = time.time()
        json_path = os.path.join(_BATCH_JSON_DIR.value, json_file)
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        
        # Look for matching H5 file
        h5_file = f"{base_name}.h5"
        h5_path = os.path.join(_BATCH_H5_DIR.value, h5_file)
        
        if not os.path.exists(h5_path):
            print(f"Warning: No matching H5 file found for {json_file}. Skipping...")
            continue
        
        print(f"Processing {i+1}/{len(batch_json_files)}: {base_name}")
        
        # Load the fold input
        fold_inputs = folding_input.load_fold_inputs_from_path(pathlib.Path(json_path))
        
        # Process each fold input - now using compiled model
        for fold_input in fold_inputs:
            output_subdir = os.path.join(_OUTPUT_DIR.value, fold_input.sanitised_name())
            process_fold_input(
                fold_input=fold_input,
                data_pipeline_config=data_pipeline_config,
                model_runner=model_runner,  # Reuse the same model
                output_dir=output_subdir,
                buckets=tuple(int(bucket) for bucket in _BUCKETS.value),
                init_guess=_INIT_GUESS.value,
                path=h5_path,
                num_samples=_NUM_SAMPLES.value,
                global_ccd=global_ccd
            )
        
        file_time = time.time() - file_start_time
        print(f"Task {base_name} completed in {file_time:.2f}s")
        
        # Periodic cache cleanup (optional)
        if (i + 1) % 10 == 0:
            _global_runner.clear_cache()
        
    total_batch_time = time.time() - batch_start_time
    avg_time_per_file = total_batch_time / len(batch_json_files)
    print(f'Batch processing completed, total time: {total_batch_time:.2f} seconds')
    print(f'Average per file: {avg_time_per_file:.2f} seconds')
    
  else:
    # Single mode processing
    print(f'Processing {len(fold_inputs)} fold inputs in single mode.')
    for fold_input in fold_inputs:
      task_start_time = time.time()
      process_fold_input(
          fold_input=fold_input,
          data_pipeline_config=data_pipeline_config,
          model_runner=model_runner,
          output_dir=os.path.join(_OUTPUT_DIR.value, fold_input.sanitised_name()),
          buckets=tuple(int(bucket) for bucket in _BUCKETS.value),
          init_guess=_INIT_GUESS.value,
          path=_INIT_PATH.value,
          num_samples=_NUM_SAMPLES.value 
      )
      task_time = time.time() - task_start_time
      print(f"Task {fold_input.name} completed in {task_time:.2f}s")

  print('All processing completed successfully.')


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'output_dir',
  ])
  app.run(main)
