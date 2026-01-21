import functools
import pathlib
import jax
import jax.numpy as jnp
import haiku as hk
from typing import Callable, Optional
import numpy as np
import time
from collections import defaultdict
import gc

from alphafold3.model import features, params
from alphafold3.model.components import base_model
from alphafold3.model.diffusion import model as diffusion_model
from alphafold3.jax.attention import attention
from alphafold3.common import base_config
from alphafold3.model.components import utils

class OptimizedModelRunner:
    """Correct version - separately compile different model versions"""
    
    def __init__(self, model_class, config, device, model_dir):
        self.model_class = model_class
        self.config = config
        self.device = device
        self.model_dir = model_dir
        
        # Load parameters
        self._model_params = params.get_model_haiku_params(model_dir=model_dir)
        
        # Key: separately compile for different init_guess values
        self._prepare_compiled_models()
        
        # Performance statistics
        self.inference_count = 0
        self.total_inference_time = 0
        self.first_inference_done = False
        
        # Position data cache
        self._position_cache = {}
    
    def _prepare_compiled_models(self):
        """Separately prepare compiled versions for init_guess=True/False"""
        
        # Version without init_guess
        @hk.transform
        def forward_fn_no_init(batch, num_samples=5):
            model = self.model_class(self.config)
            result = model(batch, init_guess=False, num_samples=num_samples)
            result['__identifier__'] = self._model_params['__meta__']['__identifier__']
            return result
        
        # Version with init_guess - accepts preloaded position arrays
        @hk.transform  
        def forward_fn_with_init(batch, init_positions, num_samples=5):
            model = self.model_class(self.config)
            
            # Temporarily replace the loading function
            import alphafold3.model.diffusion.model as dm
            original_load = dm.load_traced_array
            
            def mock_load_traced_array(path):
                return init_positions, 0, {}
            
            dm.load_traced_array = mock_load_traced_array
            
            try:
                result = model(batch, init_guess=True, num_samples=num_samples, path="dummy")
            finally:
                dm.load_traced_array = original_load
            
            result['__identifier__'] = self._model_params['__meta__']['__identifier__']
            return result
        
        # Compile both versions separately
        self._compiled_apply_no_init = jax.jit(
            forward_fn_no_init.apply,
            device=self.device,
            static_argnames=('num_samples',)
        )
        
        self._compiled_apply_with_init = jax.jit(
            forward_fn_with_init.apply,
            device=self.device,
            static_argnames=('num_samples',)
        )
        
        self._forward_fn_no_init = forward_fn_no_init
        self._forward_fn_with_init = forward_fn_with_init
    
    def _load_and_cache_positions(self, path: str):
        """Load and cache position data"""
        if path not in self._position_cache:
            from alphafold3.model.diffusion.model import load_traced_array
            loaded_array, seq_length, metadata = load_traced_array(path)
            self._position_cache[path] = jax.device_put(loaded_array, self.device)
        return self._position_cache[path]
    
    @property
    def model_params(self):
        return self._model_params
    
    def run_inference(self, 
                      featurised_example: features.BatchDict,
                      rng_key: jnp.ndarray,
                      init_guess: bool = True,
                      path: str = '',
                      num_samples: int = 5) -> base_model.ModelResult:
        """Run inference - use the correct compiled version"""
        
        start_time = time.time()
        
        # Data preprocessing
        featurised_example = jax.device_put(
            jax.tree_util.tree_map(
                jnp.asarray, utils.remove_invalidly_typed_feats(featurised_example)
            ),
            self.device,
        )
        
        if not self.first_inference_done:
            print("First inference - triggering JAX compilation...")
        
        # Choose different compiled versions based on init_guess
        if init_guess and path:
            # Load position data
            init_positions = self._load_and_cache_positions(path)
            
            # Use compiled version with init_guess
            result = self._compiled_apply_with_init(
                self._model_params,
                rng_key,
                featurised_example,
                init_positions,
                num_samples
            )
        else:
            # Use compiled version without init_guess
            result = self._compiled_apply_no_init(
                self._model_params,
                rng_key,
                featurised_example,
                num_samples
            )
        
        # Post-processing
        result = jax.tree.map(np.asarray, result)
        result = jax.tree.map(
            lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
            result,
        )
        result['__identifier__'] = result['__identifier__'].tobytes()
        
        inference_time = time.time() - start_time
        self.inference_count += 1
        self.total_inference_time += inference_time
        
        if not self.first_inference_done:
            print(f"First inference completed (including compilation time): {inference_time:.2f} seconds")
            self.first_inference_done = True
        else:
            print(f"Inference #{self.inference_count} completed: {inference_time:.2f} seconds")
        
        return result
    
    def extract_structures(self,
                           batch: features.BatchDict,
                           result: base_model.ModelResult,
                           target_name: str) -> list[base_model.InferenceResult]:
        """Extract structures"""
        return list(
            self.model_class.get_inference_result(
                batch=batch, result=result, target_name=target_name
            )
        )
    
    def clear_cache(self):
        """Clear cache"""
        print("Clearing position data cache...")
        self._position_cache.clear()
        
        print("Clearing JAX compilation cache...")
        # Clear JAX cache (optional, usually not needed)
        # jax.clear_backends()
        
        # Garbage collection
        gc.collect()
        print("Cache cleanup completed")

# Global instance
_global_runner = None

def get_optimized_runner(model_class, config, device, model_dir):
    """Get or create global optimized model runner"""
    global _global_runner
    if _global_runner is None:
        print("Initializing global model manager...")
        start_time = time.time()
        _global_runner = OptimizedModelRunner(model_class, config, device, model_dir)
        print(f"Global model initialization completed, time taken: {time.time() - start_time:.2f} seconds")
    return _global_runner