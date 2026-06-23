"""
HyperQuant — E8 Lattice Quantization for LLMs and Diffusion Transformers.

Project page : https://moonmath.ai/hyperquant/
Paper        : https://arxiv.org/abs/2606.23406
"""
from hyperquant.lattice import (
    quantize_e8int,
    quantize_d4int,
    quantize_a2int,
    quantize_z1,
)
from hyperquant.calibration import (
    calibrate_lattice_bps_to_snr,
)
from hyperquant.quant_utils import (
    lattice_alpha,
)
from hyperquant.weight_quant import (
    MMA_DTYPES,
    install_fp8_path,
    quantize_weight_for_fp8_path,
    chunked_hadamard,
    simulate_per_tile_fp8,
    simulate_per_tile_quant,
    Fp8LatticeConfig,
)
from hyperquant.kv_quant import (
    KVQuantConfig,
    install_kv_cache_quant_path,
    remove_kv_hooks,
    measured_bits_per_scalar,
    rotate_dither_lattice_pseudo_quantize,
    install_kv_residual_quant,
    remove_kv_residual_quant,
    measured_bps_residual,
)

__version__ = "0.1.0"
__all__ = [
    # Core lattice quantizers
    "quantize_e8int", "quantize_d4int", "quantize_a2int", "quantize_z1",
    # Calibration
    "calibrate_lattice_bps_to_snr", "lattice_alpha",
    # Weight quantization (use integrations.llama.convert_linears for models)
    "MMA_DTYPES",
    "install_fp8_path", "quantize_weight_for_fp8_path",
    "chunked_hadamard", "simulate_per_tile_fp8", "simulate_per_tile_quant",
    "Fp8LatticeConfig",
    # KV-cache quantization
    "KVQuantConfig", "install_kv_cache_quant_path",
    "remove_kv_hooks", "measured_bits_per_scalar",
    "rotate_dither_lattice_pseudo_quantize",
    "install_kv_residual_quant", "remove_kv_residual_quant",
    "measured_bps_residual",
]
