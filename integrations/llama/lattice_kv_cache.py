"""
LatticeQuantizedCache: int8-resident KV cache with E8 lattice quantization.

Instead of the pseudo-quant approach (quantize-dequantize → store bf16), this
stores past K and V as int8 E8 lattice codes plus float16 per-vector L2 norms:

    stored bytes ≈ 1 byte/scalar (int8 codes)
                 + 2 bytes / head_dim scalars (fp16 norms)

vs bf16's 2 bytes/scalar → ~1.97× actual GPU memory savings at head_dim=128.

The quantization noise is equivalent to the pseudo-quant hooks (same rotation,
same alpha, same lattice): the quality results remain unchanged.

Usage:
    cfg = KVQuantConfig(lattice="e8int", snr_db=..., rotation_kind="qjl")
    cache = build_lattice_kv_cache(model, cfg)
    out = model(ids, past_key_values=cache, use_cache=True)
    # out.past_key_values is the same cache object, now populated
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer

from hyperquant.quant_utils import _LATTICE_INFO, lattice_alpha
from hyperquant.kv_quant import KVQuantConfig, _generate_rotation_per_layer

_e8_quantize = _LATTICE_INFO["e8int"][2]   # quantize_e8int(x: bfloat16) → int32


def _apply_rotation_flat(
    x: torch.Tensor,
    rotation: torch.Tensor | None,
    rotation_kind: str,
    *,
    inverse: bool,
) -> torch.Tensor:
    """Apply (or invert) rotation to x [N, D] in float32."""
    if rotation is None or rotation_kind == "none":
        return x
    rot = rotation.to(device=x.device, dtype=x.dtype)
    if rotation_kind == "signs":
        return x * rot
    if rotation_kind == "qjl":
        # forward: x @ R.T  (lattice sees rotated frame)
        # inverse: x @ R
        return x @ (rot if inverse else rot.T)
    raise ValueError(f"unknown rotation_kind {rotation_kind!r}")


class LatticeQuantizedLayer(DynamicLayer):
    """Stores K/V as int8 (alpha-scaled E8 lattice codes in the rotated frame)
    plus float16 per-vector L2 norms.  Inherits get_mask_sizes from DynamicLayer.

    Memory breakdown per scalar (head_dim = 128):
      int8 code:  1.000 byte/scalar
      fp16 norm:  2 / 128 = 0.016 byte/scalar
      total:      ~1.016 byte/scalar   (vs bf16: 2 byte/scalar)
    """

    is_sliding = False

    def __init__(
        self,
        rotation_k: torch.Tensor | None,
        rotation_v: torch.Tensor | None,
        rotation_kind: str,
        alpha: float,
    ):
        super().__init__()
        self.rotation_k = rotation_k
        self.rotation_v = rotation_v
        self.rotation_kind = rotation_kind
        self.alpha = float(alpha)
        # Accumulated quantized storage; grown by cat on each update().
        self._codes_k: torch.Tensor | None = None   # [B, H, T, D] int8
        self._codes_v: torch.Tensor | None = None
        self._norms_k: torch.Tensor | None = None   # [B, H, T] float16
        self._norms_v: torch.Tensor | None = None
        # Mark initialized so DynamicLayer.update() doesn't call lazy_initialization.
        self.is_initialized = True

    # ------------------------------------------------------------------ quant
    def _quantize(
        self, x: torch.Tensor, rotation: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize [B, H, T, D] → (int8 codes [B,H,T,D], fp16 norms [B,H,T])."""
        B, H, T, D = x.shape
        x_flat = x.reshape(-1, D).float()

        x_rot = _apply_rotation_flat(x_flat, rotation, self.rotation_kind, inverse=False)

        # Per-vector L2 norm; store raw norm (not divided by sqrt(D)).
        raw_norms = x_rot.norm(dim=-1, keepdim=True).clamp_(min=1e-12)
        # u has per-scalar variance ≈ 1, matching the calibration convention.
        u = (x_rot / raw_norms) * math.sqrt(D)

        # E8 quantize: input bfloat16 matches the calibration regime.
        codes = _e8_quantize((self.alpha * u).bfloat16())  # [N, D] int32
        codes_i8 = codes.clamp_(-127, 127).to(torch.int8).reshape(B, H, T, D)

        norms_f16 = raw_norms.squeeze(-1).reshape(B, H, T).to(torch.float16)
        return codes_i8, norms_f16

    # ---------------------------------------------------------------- dequant
    def _dequantize(
        self,
        codes: torch.Tensor,
        norms: torch.Tensor,
        rotation: torch.Tensor | None,
    ) -> torch.Tensor:
        """Dequantize (int8 [B,H,T,D], fp16 norms [B,H,T]) → bf16 [B,H,T,D]."""
        B, H, T, D = codes.shape
        # u_hat = codes / alpha; x_rot_hat = u_hat / sqrt(D) * raw_norms
        u_hat = codes.float() * (1.0 / self.alpha)
        x_rot_hat = (u_hat * (1.0 / math.sqrt(D))) * norms.float().unsqueeze(-1)

        x_flat = _apply_rotation_flat(
            x_rot_hat.reshape(-1, D), rotation, self.rotation_kind, inverse=True
        )
        return x_flat.reshape(B, H, T, D).bfloat16()

    # --------------------------------------------------------------- Cache API
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codes_k, norms_k = self._quantize(key_states, self.rotation_k)
        codes_v, norms_v = self._quantize(value_states, self.rotation_v)

        if self._codes_k is None:
            self._codes_k, self._norms_k = codes_k, norms_k
            self._codes_v, self._norms_v = codes_v, norms_v
        else:
            self._codes_k = torch.cat([self._codes_k, codes_k], dim=2)
            self._norms_k = torch.cat([self._norms_k, norms_k], dim=2)
            self._codes_v = torch.cat([self._codes_v, codes_v], dim=2)
            self._norms_v = torch.cat([self._norms_v, norms_v], dim=2)

        keys = self._dequantize(self._codes_k, self._norms_k, self.rotation_k)
        values = self._dequantize(self._codes_v, self._norms_v, self.rotation_v)
        return keys, values

    def get_seq_length(self) -> int:
        if self._codes_k is None:
            return 0
        return self._codes_k.shape[2]

    def stored_bytes(self) -> int:
        """Actual GPU bytes for K and V in the compressed format."""
        b = 0
        for t in (self._codes_k, self._codes_v):
            if t is not None:
                b += t.numel()          # int8 = 1 byte each
        for t in (self._norms_k, self._norms_v):
            if t is not None:
                b += t.numel() * 2      # float16 = 2 bytes each
        return b


class LatticeQuantizedCache(DynamicCache):
    """DynamicCache subclass backed by LatticeQuantizedLayer instances.

    Constructed via build_lattice_kv_cache(model, cfg).  Pass as
    past_key_values to model() — the model calls update() on each layer
    and the cache stores/retrieves compressed int8 K/V transparently.
    """

    def __init__(self, layer_objects: list[LatticeQuantizedLayer]):
        # Bypass DynamicCache.__init__ (which sets layer_class_to_replicate=
        # DynamicLayer).  Go straight to Cache.__init__ with our pre-built
        # layer list.
        Cache.__init__(self, layers=layer_objects)

    def stored_bytes(self) -> int:
        """Total compressed bytes across all layers and K/V."""
        return sum(
            layer.stored_bytes()
            for layer in self.layers
            if isinstance(layer, LatticeQuantizedLayer)
        )


def build_lattice_kv_cache(model: nn.Module, cfg: KVQuantConfig) -> LatticeQuantizedCache:
    """Build a LatticeQuantizedCache matched to `model`'s attention layers.

    Generates per-layer K/V rotation matrices using the same seeding as
    install_kv_cache_quant_path, so the quantization noise is equivalent.
    """
    alpha = lattice_alpha(cfg.snr_db, cfg.lattice)
    layer_objects: list[LatticeQuantizedLayer] = []
    layer_idx = 0

    for _name, module in model.named_modules():
        if not (
            hasattr(module, "k_proj")
            and hasattr(module, "v_proj")
            and isinstance(module.k_proj, nn.Linear)
            and isinstance(module.v_proj, nn.Linear)
        ):
            continue

        head_dim = getattr(module, "head_dim", cfg.head_dim) or cfg.head_dim
        device = module.k_proj.weight.device
        dtype = module.k_proj.weight.dtype

        rot_k = _generate_rotation_per_layer(
            cfg.rotation_kind, head_dim, cfg.seed + 2 * layer_idx,
            device=device, dtype=dtype,
        )
        rot_v = _generate_rotation_per_layer(
            cfg.rotation_kind, head_dim, cfg.seed + 2 * layer_idx + 1,
            device=device, dtype=dtype,
        )

        layer_objects.append(
            LatticeQuantizedLayer(
                rotation_k=rot_k,
                rotation_v=rot_v,
                rotation_kind=cfg.rotation_kind,
                alpha=alpha,
            )
        )
        layer_idx += 1

    return LatticeQuantizedCache(layer_objects)


# ─────────────────────────────────────────────────────────────────────────────
# Rice-coded KV cache — variable-length storage at ~4× compression
# ─────────────────────────────────────────────────────────────────────────────

def _get_ext():
    """Lazy-import the MetalRice extension (avoids circular import)."""
    from integrations.llama.rice_linear import get_ext
    return get_ext()


class RiceKVCacheLayer(DynamicLayer):
    """KV cache layer backed by Rice-coded variable-length bitstreams.

    Memory breakdown per scalar (head_dim = 128, 4 bps Rice):
      Rice bitstream:  ~0.500 byte/scalar
      fp16 norms:       0.016 byte/scalar
      total:           ~0.516 byte/scalar   (vs bf16: 2 byte/scalar → ~3.9×)

    Design (consolidated bitstream)
    --------------------------------
    Tokens buffer in a normalized bf16 buffer until ``chunk_tokens =
    sps // head_dim`` tokens accumulate (4 tokens for sps=512, head_dim=128).
    That chunk is Rice-encoded and *appended* to a single growing words
    tensor per K/V role, keeping a globally-consistent bit-offset table.

    The key optimisation over a list-of-chunk design is that decoding the
    full history requires only **one** ``rice_decode_into`` kernel call
    (plus one rotation GEMM) instead of one call *per chunk*, eliminating
    ~3 000 kernel-launch overheads per model decode step.

    Additionally, the ``attn_data()`` method exposes the raw bitstream
    buffers so the fused decode+attention kernel (``rice_attention.py``)
    can read Rice bits and compute attention without ever materialising a
    bf16 K/V scratch tensor in global memory.
    """

    is_sliding = False

    def __init__(
        self,
        rotation_k: torch.Tensor | None,
        rotation_v: torch.Tensor | None,
        rotation_kind: str,
        alpha: float,
        rice_k: int = 2,
        sps: int = 512,
        head_dim: int = 128,
        n_kv_heads: int = 8,
    ):
        super().__init__()
        self.rotation_k    = rotation_k
        self.rotation_v    = rotation_v
        self.rotation_kind = rotation_kind
        self.alpha         = float(alpha)
        self.rice_k        = int(rice_k)
        self.sps           = int(sps)
        self.head_dim      = int(head_dim)
        self.n_kv_heads    = int(n_kv_heads)
        self.chunk_tokens  = sps // head_dim   # = 4 for sps=512, head_dim=128

        # ── Consolidated bitstream (grows via torch.cat on each new chunk).
        # Bit offsets in _offsets_* are GLOBAL (relative to words[0][bit 0]).
        self._words_k:   torch.Tensor | None = None   # [total_words] uint32
        self._offsets_k: torch.Tensor | None = None   # [total_streams] uint32
        self._ks_k:      torch.Tensor | None = None   # [total_streams] uint8
        self._norms_k:   torch.Tensor | None = None   # [H, n_complete] float16
        self._words_v:   torch.Tensor | None = None
        self._offsets_v: torch.Tensor | None = None
        self._ks_v:      torch.Tensor | None = None
        self._norms_v:   torch.Tensor | None = None

        # ── Pending buffer (normalized bf16, in the rotated frame).
        # Shape: [n_kv_heads, n_pending, head_dim]
        self._pend_uk: torch.Tensor | None = None
        self._pend_uv: torch.Tensor | None = None
        self._pend_nk: torch.Tensor | None = None   # raw L2 norms [H, n_pend]
        self._pend_nv: torch.Tensor | None = None
        self._n_pending  = 0
        self._n_complete = 0

        self.is_initialized = True

    # ──────────────────────────────────────────── helpers
    def _rot(self, x_flat: torch.Tensor, rotation, inverse: bool) -> torch.Tensor:
        return _apply_rotation_flat(x_flat, rotation, self.rotation_kind,
                                    inverse=inverse)

    def _rotate_norm(self, x: torch.Tensor, rotation):
        """x: [B, H, T, D] → (u [H,T,D] bf16, norms [H,T] fp16)."""
        B, H, T, D = x.shape
        flat = x.reshape(-1, D).float()
        rot  = self._rot(flat, rotation, inverse=False)
        raw  = rot.norm(dim=-1, keepdim=True).clamp_(min=1e-12)
        u    = (rot / raw) * math.sqrt(D)
        return u.reshape(H, T, D).bfloat16(), raw.squeeze(-1).reshape(H, T).half()

    # ──────────────────────────────────────────── encode one chunk
    def _encode_chunk(self, u: torch.Tensor):
        """Rice-encode [H, CT, D] normalized bf16 → (words, offsets, ks, ns, nw)."""
        H, T, D = u.shape
        flat_cpu = u.reshape(-1, D).cpu().bfloat16().contiguous()
        words, offsets, ks, ns, nw, tot = _get_ext().rice_encode(
            flat_cpu, self.alpha, self.rice_k, self.sps)
        assert tot == H * T * D
        return words, offsets, ks, ns, nw

    def _append_chunk(self, words_new, offsets_new, ks_new, norms_chunk,
                      role: str):
        """Append a newly-encoded chunk to the consolidated bitstream."""
        if role == 'k':
            w, o, k, n = self._words_k, self._offsets_k, self._ks_k, self._norms_k
        else:
            w, o, k, n = self._words_v, self._offsets_v, self._ks_v, self._norms_v

        # Shift new offsets by the bit length of the existing words buffer.
        bit_shift = (0 if w is None else w.numel() * 32)
        offsets_global = offsets_new + bit_shift

        words_new_  = torch.cat([w, words_new  ]) if w is not None else words_new
        offsets_new_= torch.cat([o, offsets_global]) if o is not None else offsets_global
        ks_new_     = torch.cat([k, ks_new     ]) if k is not None else ks_new
        norms_new_  = torch.cat([n, norms_chunk ], dim=1) if n is not None else norms_chunk

        if role == 'k':
            self._words_k, self._offsets_k, self._ks_k, self._norms_k = (
                words_new_, offsets_new_, ks_new_, norms_new_)
        else:
            self._words_v, self._offsets_v, self._ks_v, self._norms_v = (
                words_new_, offsets_new_, ks_new_, norms_new_)

    # ──────────────────────────────────────────── decode full history
    def _decode_full(self, role: str) -> torch.Tensor:
        """Decode all complete Rice-coded tokens → [H, n_complete, D] bf16."""
        H, D = self.n_kv_heads, self.head_dim
        T    = self._n_complete
        if T == 0:
            dev = (self._words_k if role == 'k' else self._words_v).device
            return torch.zeros(H, 0, D, dtype=torch.bfloat16, device=dev)

        if role == 'k':
            words, offsets, ks = self._words_k, self._offsets_k, self._ks_k
            norms, rotation    = self._norms_k, self.rotation_k
        else:
            words, offsets, ks = self._words_v, self._offsets_v, self._ks_v
            norms, rotation    = self._norms_v, self.rotation_v

        ns   = offsets.numel()
        nw   = H * T * D // 8                    # total E8 words
        dev  = words.device
        scratch = torch.empty(nw * 8, dtype=torch.bfloat16, device=dev)
        _get_ext().rice_decode_into(words, offsets, ks, ns, self.sps, nw,
                                    self.alpha, scratch)
        # The consolidated bitstream is chunk-major:
        # scratch layout = [n_chunks, H, chunk_tokens, D] (flat).
        # Permute to [H, T, D] before applying norms/rotation.
        n_chunks  = T // self.chunk_tokens
        u_hat_raw = scratch.view(n_chunks, H, self.chunk_tokens, D).float()
        u_hat     = u_hat_raw.permute(1, 0, 2, 3).contiguous().view(H, T, D)
        x_rot_h   = u_hat * (norms.float().unsqueeze(-1) / math.sqrt(D))
        x_h       = self._rot(x_rot_h.reshape(-1, D), rotation, inverse=True)
        return x_h.reshape(H, T, D).bfloat16()

    # ──────────────────────────────────────────── DynamicLayer API
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V, encode complete chunks, decode full history."""
        B, H, T_new, D = key_states.shape

        uk, nk = self._rotate_norm(key_states,   self.rotation_k)
        uv, nv = self._rotate_norm(value_states, self.rotation_v)

        # Append to pending buffer.
        if self._pend_uk is None:
            self._pend_uk, self._pend_nk = uk, nk
            self._pend_uv, self._pend_nv = uv, nv
        else:
            self._pend_uk = torch.cat([self._pend_uk, uk], dim=1)
            self._pend_nk = torch.cat([self._pend_nk, nk], dim=1)
            self._pend_uv = torch.cat([self._pend_uv, uv], dim=1)
            self._pend_nv = torch.cat([self._pend_nv, nv], dim=1)
        self._n_pending += T_new

        # Encode complete chunks into consolidated bitstream.
        CT = self.chunk_tokens
        while self._n_pending >= CT:
            cuk = self._pend_uk[:, :CT, :].contiguous()
            cnk = self._pend_nk[:, :CT]
            cuv = self._pend_uv[:, :CT, :].contiguous()
            cnv = self._pend_nv[:, :CT]

            wk, ok, kk, _, _ = self._encode_chunk(cuk)
            wv, ov, kv, _, _ = self._encode_chunk(cuv)
            self._append_chunk(wk, ok, kk, cnk, 'k')
            self._append_chunk(wv, ov, kv, cnv, 'v')

            self._pend_uk = self._pend_uk[:, CT:, :]
            self._pend_nk = self._pend_nk[:, CT:]
            self._pend_uv = self._pend_uv[:, CT:, :]
            self._pend_nv = self._pend_nv[:, CT:]
            self._n_pending  -= CT
            self._n_complete += CT

        # Decode complete history (ONE kernel call each for K and V).
        keys_parts: list[torch.Tensor] = []
        vals_parts: list[torch.Tensor] = []
        if self._n_complete > 0:
            keys_parts.append(self._decode_full('k'))
            vals_parts.append(self._decode_full('v'))

        # Reconstruct pending (undo normalisation + rotation).
        if self._n_pending > 0:
            x_rk = (self._pend_uk.float() *
                    (self._pend_nk.float().unsqueeze(-1) / math.sqrt(D)))
            x_rv = (self._pend_uv.float() *
                    (self._pend_nv.float().unsqueeze(-1) / math.sqrt(D)))
            keys_parts.append(self._rot(x_rk.reshape(-1, D),
                                        self.rotation_k, inverse=True)
                              .reshape(H, self._n_pending, D).bfloat16())
            vals_parts.append(self._rot(x_rv.reshape(-1, D),
                                        self.rotation_v, inverse=True)
                              .reshape(H, self._n_pending, D).bfloat16())

        keys = torch.cat(keys_parts, dim=1).unsqueeze(0)  # [1, H, T_total, D]
        vals = torch.cat(vals_parts, dim=1).unsqueeze(0)
        return keys, vals

    # ──────────────────────────────────────────── fused attention API
    def attn_data(self) -> dict:
        """Return all raw buffers needed by the fused decode+attention kernel.

        The fused kernel reads the Rice bitstream directly and computes
        attention without ever materialising a bf16 K/V scratch tensor.
        Returns a dict with keys: words_k, offsets_k, ks_k, norms_k,
        words_v, offsets_v, ks_v, norms_v, pend_uk, pend_nk, pend_uv,
        pend_nv, n_complete, n_pending, chunk_tokens, sps, alpha,
        n_kv_heads, head_dim, rotation_k, rotation_v, rotation_kind.
        """
        return dict(
            words_k=self._words_k, offsets_k=self._offsets_k,
            ks_k=self._ks_k, norms_k=self._norms_k,
            words_v=self._words_v, offsets_v=self._offsets_v,
            ks_v=self._ks_v, norms_v=self._norms_v,
            pend_uk=self._pend_uk, pend_nk=self._pend_nk,
            pend_uv=self._pend_uv, pend_nv=self._pend_nv,
            n_complete=self._n_complete, n_pending=self._n_pending,
            chunk_tokens=self.chunk_tokens, sps=self.sps, alpha=self.alpha,
            n_kv_heads=self.n_kv_heads, head_dim=self.head_dim,
            rotation_k=self.rotation_k, rotation_v=self.rotation_v,
            rotation_kind=self.rotation_kind,
        )

    def get_seq_length(self) -> int:
        return self._n_complete + self._n_pending

    def stored_bytes(self) -> int:
        b = 0
        for t in (self._words_k, self._words_v):
            if t is not None: b += t.numel() * 4
        for t in (self._offsets_k, self._offsets_v):
            if t is not None: b += t.numel() * 4
        for t in (self._ks_k, self._ks_v):
            if t is not None: b += t.numel()
        for t in (self._norms_k, self._norms_v):
            if t is not None: b += t.numel() * 2
        for t in (self._pend_uk, self._pend_uv, self._pend_nk, self._pend_nv):
            if t is not None: b += t.numel() * 2
        return b


class RiceQuantizedCache(DynamicCache):
    """DynamicCache backed by RiceKVCacheLayer (variable-length Rice storage)."""

    def __init__(self, layer_objects: list[RiceKVCacheLayer]):
        Cache.__init__(self, layers=layer_objects)

    def stored_bytes(self) -> int:
        return sum(
            layer.stored_bytes()
            for layer in self.layers
            if isinstance(layer, RiceKVCacheLayer)
        )


def build_rice_kv_cache(
    model: nn.Module,
    cfg: KVQuantConfig,
    rice_k: int = 2,
    sps: int = 512,
) -> RiceQuantizedCache:
    """Build a RiceQuantizedCache matched to ``model``'s attention layers.

    Uses the same rotation seeding as :func:`build_lattice_kv_cache` so the
    quantization noise is equivalent.  Storage is variable-length Rice
    (``~0.516`` bytes/scalar at 4 bps) instead of int8 (1.016 bytes/scalar).
    """
    alpha = lattice_alpha(cfg.snr_db, cfg.lattice)
    layer_objects: list[RiceKVCacheLayer] = []
    layer_idx = 0

    for _name, module in model.named_modules():
        if not (
            hasattr(module, "k_proj")
            and hasattr(module, "v_proj")
            and isinstance(module.k_proj, nn.Linear)
            and isinstance(module.v_proj, nn.Linear)
        ):
            continue

        head_dim = getattr(module, "head_dim", cfg.head_dim) or cfg.head_dim
        n_kv_heads = module.k_proj.out_features // head_dim
        device = module.k_proj.weight.device
        dtype  = module.k_proj.weight.dtype

        rot_k = _generate_rotation_per_layer(
            cfg.rotation_kind, head_dim, cfg.seed + 2 * layer_idx,
            device=device, dtype=dtype,
        )
        rot_v = _generate_rotation_per_layer(
            cfg.rotation_kind, head_dim, cfg.seed + 2 * layer_idx + 1,
            device=device, dtype=dtype,
        )

        layer_objects.append(
            RiceKVCacheLayer(
                rotation_k=rot_k,
                rotation_v=rot_v,
                rotation_kind=cfg.rotation_kind,
                alpha=alpha,
                rice_k=rice_k,
                sps=sps,
                head_dim=head_dim,
                n_kv_heads=n_kv_heads,
            )
        )
        layer_idx += 1

    return RiceQuantizedCache(layer_objects)
