"""
Example: quantize a HuggingFace model's linear weights with HyperQuant.

This replaces every nn.Linear (except lm_head / embeddings) with a
RiceLinear that stores weights as a Rice-coded bitstream at ~4 bps,
giving ~3.9× weight-memory reduction and near-lossless quality.

Usage:
    python examples/quantize_weights.py --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hyperquant import calibrate_lattice_bps_to_snr, lattice_alpha
from integrations.llama.rice_linear import convert_linears

GiB = 1024 ** 3


def quantize_model(model_name: str, bps: float = 4.0):
    print(f"Loading {model_name} in bf16 ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    resident_before = torch.cuda.memory_allocated() / GiB
    print(f"Resident before quantization: {resident_before:.2f} GiB")

    # Calibrate: map target bps → E8 lattice alpha (scale parameter)
    snr_db = calibrate_lattice_bps_to_snr(
        lattices=["e8int"], target_bps_list=[bps]
    )["e8int"][bps]["snr_db"]
    alpha = lattice_alpha(snr_db, "e8int")
    print(f"Target {bps} bps → SNR {snr_db:.2f} dB, alpha = {alpha:.3f}")

    # Replace all nn.Linear weights (mma="int8" default: cublasLt INT8 IMMA GEMM).
    stats = convert_linears(
        model, skip=("lm_head",), alpha=alpha, verbose=True
    )
    torch.cuda.empty_cache()

    resident_after = torch.cuda.memory_allocated() / GiB
    print(f"\nConverted {stats['n_converted']} layers")
    print(f"Weight memory: {stats['orig_weight_bytes']/GiB:.2f} → "
          f"{stats['compressed_bytes']/GiB:.2f} GiB "
          f"({stats['compression_x']:.2f}×)")
    print(f"Resident after quantization: {resident_after:.2f} GiB "
          f"({resident_before / resident_after:.2f}× reduction)")

    # Quick quality check on a short prompt
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    print(f"\nGeneration check: {tokenizer.decode(out[0])}")

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--bps", type=float, default=4.0)
    args = parser.parse_args()
    quantize_model(args.model, args.bps)
