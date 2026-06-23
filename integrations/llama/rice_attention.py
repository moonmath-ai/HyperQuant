"""
rice_attention.py — LlamaAttention monkey-patch for the Rice-coded KV cache.

Replaces the standard three-step path:
  (1) rice_decode_into  → bf16 K scratch  (n_chunks kernel calls)
  (2) rotation GEMM     → bf16 K history  (n_chunks GEMMs)
  (3) Q·K^T, softmax, ·V  (standard attention)

with a single kernel call per attention layer that reads the Rice bitstream
and computes the softmax-weighted output without any bf16 K/V scratch tensor.

For prefill (T_q > 1) or when T_total > ATTN_T_MAX, the patched forward
falls back to the standard two-kernel path (consolidate decode → standard
attention), which still benefits from the consolidated bitstream optimization
(one decode call per role vs. n_chunks calls).

Usage:
    cache = build_rice_kv_cache(model, cfg)
    install_rice_attention_hooks(model, cache)
    # ... run model ...
    remove_rice_attention_hooks(model)
"""
from __future__ import annotations

import math
import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_HANDLES: list = []   # storage for (module, orig_forward) pairs


def _apply_rotation(x_flat: torch.Tensor, rotation, rotation_kind: str,
                    inverse: bool) -> torch.Tensor:
    if rotation is None or rotation_kind == "none":
        return x_flat
    rot = rotation.to(device=x_flat.device, dtype=x_flat.dtype)
    if rotation_kind == "signs":
        return x_flat * rot
    if rotation_kind == "qjl":
        # forward: x @ R.T;  inverse: x @ R
        return x_flat @ (rot if inverse else rot.T)
    raise ValueError(f"unknown rotation_kind {rotation_kind!r}")


def _fused_attn_for_layer(
    ext,
    Q_postRoPE: torch.Tensor,   # [B, n_q, 1, D]  bf16, after RoPE
    attn_layer,                  # the RiceKVCacheLayer for this transformer layer
) -> torch.Tensor | None:
    """Attempt fused Rice decode + attention for single decode step.

    Returns [B, 1, n_q_heads * D] bf16 attn_output, or None on fallback.
    """
    D = Q_postRoPE.shape[-1]
    B, n_q, T_q, _ = Q_postRoPE.shape
    if T_q != 1:
        return None

    data = attn_layer.attn_data()
    n_kv  = data["n_kv_heads"]
    n_qpk = n_q // n_kv
    n_complete = data["n_complete"]
    n_pending  = data["n_pending"]
    T_total    = n_complete + n_pending

    if T_total == 0 or data["words_k"] is None:
        return None

    t_max = ext.rice_fused_attn_t_max()
    if T_total > t_max:
        return None  # caller falls back

    # ── Pre-rotate Q: Q_prerot = Q @ R_k  (maps Q into K's rotated frame).
    # Since rotation is shared across all KV heads of this layer, one batch
    # GEMM replaces a Python loop over 32 Q heads.
    Q_flat = Q_postRoPE.squeeze(2).squeeze(0).float()   # [n_q, D]
    rot_k  = data["rotation_k"]    # [D, D]
    rot_v  = data["rotation_v"]    # [D, D]
    rk     = data["rotation_kind"]

    # K_rot = K @ R.T  (forward rotation, inverse=False).
    # For Q·K = Q_prerot·K_rot, need Q_prerot = Q @ R.T (same forward rotation).
    if rk == "qjl":
        Q_prerot = (Q_flat @ rot_k.float().T).bfloat16()        # [n_q, D]
    elif rk == "signs":
        Q_prerot = (Q_flat * rot_k.float()).bfloat16()           # signs are self-inverse
    else:
        Q_prerot = Q_flat.bfloat16()

    dev   = Q_prerot.device
    alpha = data["alpha"]
    sps   = data["sps"]
    ct    = data["chunk_tokens"]

    # Norms are stored as fp16 but kernel expects bf16 bits → convert.
    norms_k_bf16 = data["norms_k"].bfloat16().reshape(n_kv, n_complete).contiguous()
    norms_v_bf16 = data["norms_v"].bfloat16().reshape(n_kv, n_complete).contiguous()

    def safe_pend(t, h, p, d, is_norm=False):
        if t is None or p == 0:
            return torch.empty(0, dtype=torch.bfloat16, device=dev)
        return t.bfloat16().contiguous().reshape(h * p, d) if not is_norm \
               else t.bfloat16().contiguous().reshape(h * p)

    pk  = safe_pend(data["pend_uk"], n_kv, n_pending, D)
    pnk = safe_pend(data["pend_nk"], n_kv, n_pending, 1, is_norm=True)
    pv  = safe_pend(data["pend_uv"], n_kv, n_pending, D)
    pnv = safe_pend(data["pend_nv"], n_kv, n_pending, 1, is_norm=True)

    out_rot = ext.rice_fused_attn_decode(
        data["words_k"], data["offsets_k"], data["ks_k"], norms_k_bf16,
        data["words_v"], data["offsets_v"], data["ks_v"], norms_v_bf16,
        pk, pnk, pv, pnv, n_pending,
        Q_prerot,
        n_kv, n_qpk, D,
        n_complete, ct, sps, alpha,
    )

    if out_rot is None or out_rot.numel() == 0:
        return None  # kernel signalled fallback

    # out_rot: [n_q, D] float32 in V's rotated frame.
    # Apply inverse V rotation as one batch GEMM.
    out_f32 = out_rot.float()   # [n_q, D]
    if rk == "qjl":
        out_f32 = out_f32 @ rot_v.float()      # @ R  (inverse: x @ R)
    elif rk == "signs":
        out_f32 = out_f32 * rot_v.float()

    return out_f32.bfloat16().unsqueeze(0).unsqueeze(2)  # [1, n_q, 1, D]


def make_rice_attn_forward(orig_module, cache, layer_idx: int):
    """Return a patched forward for a single LlamaAttention module."""

    def patched_forward(
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        from integrations.llama.rice_linear import get_ext
        ext = get_ext()

        B, T_q, _ = hidden_states.shape
        head_dim   = orig_module.head_dim
        n_q_heads  = orig_module.num_heads
        n_kv_heads = orig_module.num_key_value_heads
        n_q_per_kv = n_q_heads // n_kv_heads

        # 1. Projections
        q = orig_module.q_proj(hidden_states).view(B, T_q, n_q_heads,  head_dim).transpose(1, 2)
        k = orig_module.k_proj(hidden_states).view(B, T_q, n_kv_heads, head_dim).transpose(1, 2)
        v = orig_module.v_proj(hidden_states).view(B, T_q, n_kv_heads, head_dim).transpose(1, 2)

        # 2. RoPE
        cos, sin = position_embeddings
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. Update Rice cache (encode new tokens; may be a RiceKVCacheLayer or None)
        rice_layer = None
        if (past_key_values is not None
                and hasattr(past_key_values, 'layers')
                and layer_idx < len(past_key_values.layers)):
            lyr = past_key_values.layers[layer_idx]
            if hasattr(lyr, 'attn_data'):
                rice_layer = lyr
                # Append new tokens to the Rice cache (encode only, no decode).
                rice_layer.update(k, v)

        # 4a. Fused decode + attention (decode step, T_q=1, fused path).
        if T_q == 1 and rice_layer is not None:
            attn_out = _fused_attn_for_layer(ext, q, rice_layer)
            if attn_out is not None:
                # attn_out: [B, n_q, 1, D]
                attn_out = attn_out.squeeze(2).reshape(B, T_q, n_q_heads * head_dim)
                return orig_module.o_proj(attn_out), None, past_key_values

        # 4b. Fallback: decode full history (consolidated, one kernel/role) + standard attn.
        if rice_layer is not None:
            k_full, v_full = rice_layer._decode_full('k'), rice_layer._decode_full('v')
            # Append pending
            if rice_layer._n_pending > 0:
                D = head_dim
                H = n_kv_heads
                x_rk = rice_layer._pend_uk.float() * (rice_layer._pend_nk.float().unsqueeze(-1) / math.sqrt(D))
                x_rv = rice_layer._pend_uv.float() * (rice_layer._pend_nv.float().unsqueeze(-1) / math.sqrt(D))
                kp = rice_layer._rot(x_rk.reshape(-1, D), rice_layer.rotation_k, True).reshape(H, rice_layer._n_pending, D).bfloat16()
                vp = rice_layer._rot(x_rv.reshape(-1, D), rice_layer.rotation_v, True).reshape(H, rice_layer._n_pending, D).bfloat16()
                k_full = torch.cat([k_full, kp], dim=1)
                v_full = torch.cat([v_full, vp], dim=1)
            k_hist = k_full.unsqueeze(0)   # [1, H, T, D]
            v_hist = v_full.unsqueeze(0)
        else:
            # Standard HF cache or no cache (prefill).
            if past_key_values is not None:
                k_hist, v_hist = past_key_values.update(k, v, layer_idx)
            else:
                k_hist, v_hist = k, v

        # GQA expansion
        from transformers.models.llama.modeling_llama import repeat_kv
        k_hist = repeat_kv(k_hist, n_q_per_kv)
        v_hist = repeat_kv(v_hist, n_q_per_kv)

        scale = 1.0 / math.sqrt(head_dim)
        attn_weights = torch.matmul(q, k_hist.transpose(-2, -1)) * scale
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :k_hist.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_out = torch.matmul(attn_weights, v_hist)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, T_q, -1)
        return orig_module.o_proj(attn_out), None, past_key_values

    return patched_forward


def install_rice_attention_hooks(model: nn.Module, cache) -> None:
    """Monkey-patch all LlamaAttention modules to use fused Rice decode+attention.

    The patched forward uses the fused CUDA kernel for single-token decode
    (T_q=1) and falls back to the consolidated-bitstream two-kernel path for
    prefill (T_q>1) or when T_total exceeds the kernel's shared-memory budget.
    """
    global _HANDLES
    _HANDLES.clear()
    layer_idx = 0
    for name, module in model.named_modules():
        if not (hasattr(module, 'q_proj') and hasattr(module, 'k_proj')
                and hasattr(module, 'head_dim')):
            continue
        orig = module.forward
        module.forward = types.MethodType(
            make_rice_attn_forward(module, cache, layer_idx), module)
        _HANDLES.append((module, orig))
        layer_idx += 1


def remove_rice_attention_hooks() -> None:
    """Restore original LlamaAttention.forward on all patched modules."""
    global _HANDLES
    for module, orig in _HANDLES:
        module.forward = orig
    _HANDLES.clear()
