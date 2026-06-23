"""
Example: quantize the KV cache of a HuggingFace model with HyperQuant.

The KV cache stores past key and value tensors as Rice-coded bitstreams
(variable-length, ~4 bps), giving ~3.8× actual GPU memory reduction per
cached token while preserving near-lossless attention quality.

Usage:
    python examples/quantize_kv_cache.py --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hyperquant.calibration import calibrate_lattice_bps_to_snr
from hyperquant.kv_quant import KVQuantConfig
from integrations.llama.lattice_kv_cache import build_rice_kv_cache

GiB = 1024 ** 3


def quantize_kv_demo(model_name: str, bps: float = 4.0, context_len: int = 512):
    print(f"Loading {model_name} in bf16 ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    # Calibrate bps → SNR
    snr_db = calibrate_lattice_bps_to_snr(
        lattices=["e8int"], target_bps_list=[bps]
    )["e8int"][bps]["snr_db"]
    kv_cfg = KVQuantConfig(
        lattice="e8int", snr_db=snr_db,
        rotation_kind="qjl",   # Haar-uniform orthogonal rotation for unbiasedness
        quantize_k=True, quantize_v=True,
    )
    print(f"KV cache @ {bps} bps (SNR {snr_db:.1f} dB, QJL rotation)")

    # --- Baseline: bf16 KV cache ---
    prompt_ids = tokenizer("Hello, " * 50, return_tensors="pt").input_ids.to("cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out_bf16 = model(prompt_ids[:, :context_len], use_cache=True)
    peak_bf16 = torch.cuda.max_memory_allocated() / GiB
    kv_bf16_bytes = sum(
        k.numel() * 2 + v.numel() * 2
        for k, v in out_bf16.past_key_values.to_legacy_cache()
    )
    print(f"\nBaseline (bf16) peak GPU: {peak_bf16:.2f} GiB, "
          f"KV cache: {kv_bf16_bytes/1024**2:.1f} MB")
    del out_bf16
    torch.cuda.empty_cache()

    # --- HyperQuant: Rice-coded KV cache ---
    rice_cache = build_rice_kv_cache(model, kv_cfg)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(prompt_ids[:, :context_len],
                  past_key_values=rice_cache, use_cache=True)
    peak_rice = torch.cuda.max_memory_allocated() / GiB
    stored = rice_cache.stored_bytes()
    print(f"HyperQuant Rice KV peak GPU: {peak_rice:.2f} GiB, "
          f"stored: {stored/1024**2:.2f} MB "
          f"({kv_bf16_bytes/stored:.2f}× compression)")

    # Generation demo with quantized KV cache
    torch.cuda.empty_cache()
    rice_cache2 = build_rice_kv_cache(model, kv_cfg)
    prompt = "The theory of relativity was developed by"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=30, do_sample=False,
            past_key_values=rice_cache2,
        )
    print(f"\nGeneration with Rice KV cache:\n  {tokenizer.decode(out[0])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--bps", type=float, default=4.0)
    parser.add_argument("--context", type=int, default=512)
    args = parser.parse_args()
    quantize_kv_demo(args.model, args.bps, args.context)
