"""
E8int, D4int, A2int, and Z1int lattice quantisation for iid N(0, I) sources.

Pipeline (high-resolution Gaussian compression at fixed SNR):

    x  ──×α──►  α·x  ──Q_Lint──►  Y  ──structural strip──►  symbols  ──Rice──►  bits

Supported lattices (best to worst granular efficiency):
    E8int = 2·E8  (8-D, default)  ≈ 3.76 bits/scalar at 21 dB SNR  (+0.146 b/sc gap)
    D4int = 2·D4  (4-D)                                              (+0.194 b/sc gap)
    A2int = 2·A2  (2-D, hexagonal)                                   (+0.227 b/sc gap)
    Z1int = Z     (1-D, nearest rounding)  — baseline, no structure  (+0.255 b/sc gap)

E8int / D4int structural coding (single Rice k):
    (A)  Strip shared-parity: coset bit c (E8int only) + halve → s_1..s_{N−1}.
    (B)  Strip sum-parity: Σ s ≡ 0 (mod 2) → t = (s_N − forced_parity) >> 1.
    (C)  combined = 2·zz(t) + c  coded with Rice k (D4int: c = 0, LSB implicit).

A2int structural coding (two independent Rice parameters k_ty, k_x):
    Stored as (n_y, n_x) with n_y + n_x ≡ 0 (mod 2); physical y = n_y·√3.
    Strip parity: p = n_x & 1;  t_y = (n_y − p) >> 1.  Saves 0.5 bit/scalar.
    Encode t_y with Rice k_ty and n_x with Rice k_x independently.

Z1int: no structural decomposition.  Rice code each rounded scalar directly.

Hard operating constraint: every quantised scalar must fit in 8 bits.
P(|y_ij| > 127) is always reported, never raises an exception.

Usage:
    python lattice_quant.py                                             # sweep 20-30 dB, E8int
    python lattice_quant.py --snr-min 21 --snr-max 21                  # single run, 21 dB
    python lattice_quant.py --lattice d4int                             # D4int sweep
    python lattice_quant.py --lattice a2int                             # A2int sweep → a2_analysis.*
    python lattice_quant.py --lattice z1int                             # Z1int sweep → z1_analysis.*
    python lattice_quant.py --help
"""

from __future__ import annotations

import lzma
import math
import os
import zlib
from typing import Iterable

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ALPHA_CAL    = 50.0        # Calibration scale: ≫ any lattice covering radius.
RICE_K_RANGE = range(8)    # Rice parameter sweep for k-selection.

# ── E8int = 2·E8 ─────────────────────────────────────────────────────────────
# G(E8) — E8 normalised second moment (Conway & Sloane, Sphere Packings, Table 2.3).
# G(L) = (1/n) · (∫_V ‖x‖² dx) / V(L)^(1+2/n),  dimensionless, scale-invariant.
G_E8     = 0.0716568
N_DIM_E8 = 8
# V(E8int) = 2^8 × V(E8) = 256 × 1 = 256;  256^(2/8) = 256^(1/4) = 4.
#   MSE_voronoi = G(L) · n · V^(2/n) = 0.0716568 × 8 × 4 ≈ 2.293
MSE_VORONOI_E8_ANALYTIC = G_E8 * N_DIM_E8 * (256 ** (2 / N_DIM_E8))   # ≈ 2.293

# ── D4int = 2·D4 ─────────────────────────────────────────────────────────────
# G(D4) — D4 normalised second moment (Conway & Sloane, Sphere Packings, Table 2.3).
# D4 is the densest 4-D lattice: { x ∈ Z^4 : Σ x_i ≡ 0 (mod 2) }.
G_D4     = 0.076603
N_DIM_D4 = 4
# V(D4) = 2 (D4 is index-2 in Z^4);  V(D4int) = 2^4 × 2 = 32;  32^(2/4) = √32.
#   MSE_voronoi = G(D4) · 4 · √32 ≈ 1.733
MSE_VORONOI_D4_ANALYTIC = G_D4 * N_DIM_D4 * (32 ** (2 / N_DIM_D4))    # ≈ 1.733

# ── A2int = 2·A2  (2-D, hexagonal lattice) ───────────────────────────────────
# G(A2) = 5√3/108  (Conway & Sloane; densest 2-D lattice / hexagonal packing).
# Physical coords: (n_y·√3, n_x).  Stored: (n_y, n_x) ∈ Z² with n_y+n_x ≡ 0 (mod 2).
# All 6 nearest neighbours at distance 2.
G_A2     = 5 * math.sqrt(3) / 108    # ≈ 0.080188  (= 5√3/108)
N_DIM_A2 = 2
SQRT3    = math.sqrt(3)
# V(A2int) = |det [[√3, 0],[0, 2]]| = 2√3;  V^(2/N) = 2√3  (N=2).
#   MSE_voronoi = G·N·V^(2/N) = (5√3/108)·2·2√3 = 60/108 = 5/9 ≈ 0.5556
MSE_VORONOI_A2_ANALYTIC = G_A2 * N_DIM_A2 * (2 * math.sqrt(3))   # = 5/9 ≈ 0.5556

# ── Z1int = Z  (1-D, integer lattice) ────────────────────────────────────────
# Nearest-integer quantisation: each scalar is rounded independently.
# G(Z1) = 1/12 (normalised second moment of the Voronoi cell [-½, ½]).
# No structural decomposition — all savings come solely from Rice coding.
G_Z1     = 1 / 12           # ≈ 0.08333
N_DIM_Z1 = 1
# V(Z1) = 1;  V^(2/N) = 1^(2/1) = 1.
#   MSE_voronoi = G·N·V^(2/N) = (1/12)·1·1 = 1/12 ≈ 0.0833
MSE_VORONOI_Z1_ANALYTIC = G_Z1   # = 1/12

# Allowed relative deviation of empirical MSE from the analytic value.
# bfloat16 rounding causes a ~0.5% systematic offset; Monte-Carlo noise ~0.1%.
# A 5% tolerance catches real bugs while accommodating both.
MSE_VORONOI_TOL = 0.05


def _rsnr_lin(rsnr_db: float) -> float:
    return 10.0 ** (rsnr_db / 10.0)


def _lattice_display(lattice: str) -> str:
    """'e8int' → 'E8int',  'd4int' → 'D4int'."""
    return lattice[:-3].upper() + "int"


# ─────────────────────────────────────────────────────────────────────────────
# Quantisers
# ─────────────────────────────────────────────────────────────────────────────

def _fix_parity(u: torch.Tensor, x_f32: torch.Tensor) -> torch.Tensor:
    """
    Force a same-parity integer vector u to satisfy Σ u ≡ 0 (mod 4).

    Used by both E8int and D4int quantisers.  Two facts justify a one-component fix:

      (i)  u is the nearest same-parity integer to x, so |u_j − x_j| ≤ 1.
           A ±2 flip costs 4 ± 4·(u_j − x_j) in squared error, in [0, 4].

      (ii) Same-parity inputs always yield Σ u ≡ 0 or 2 (mod 4), so a single
           ±2 flip is necessary and sufficient when Σ u ≡ 2 (mod 4).

    Picking j with the largest |u_j − x_j| minimises the squared-error cost;
    the flip direction is "toward x_j", i.e. corr = −2·sign(u_j − x_j).
    """
    u = u.clone()
    s = u.sum(dim=-1)
    bad = (s & 3) != 0          # fires when Σ u ≡ 2 (mod 4)
    if not bad.any():
        return u

    u_bad = u[bad]
    x_bad = x_f32[bad]
    err   = u_bad.float() - x_bad
    j     = err.abs().argmax(dim=-1)

    rows  = torch.arange(u_bad.shape[0], device=u.device)
    delta = err[rows, j]
    corr  = (-2.0 * delta.sign()).to(torch.int32)
    corr[corr == 0] = 2          # tie (delta == 0) is unreachable in practice; safe fallback.

    u_bad[rows, j] += corr
    u[bad] = u_bad
    return u


def quantize_e8int(x: torch.Tensor) -> torch.Tensor:
    """
    Nearest-point decoder for E8int = 2·E8 (Conway & Sloane, lifted by ×2).

    Try the two cosets independently:
        • Even coset: round each coord to nearest EVEN integer, then enforce Σ ≡ 0 mod 4.
        • Odd  coset: round each coord to nearest ODD  integer, then enforce Σ ≡ 0 mod 4.

    Return whichever candidate is closer in ℓ₂.  This is provably optimal: the
    nearest E8 point lies in exactly one coset, and within each coset the
    "round + parity-fix" decoder is the nearest D₈ point of that coset.

    Args:
        x: (N, 8) bf16 — already-α-scaled input.
    Returns:
        (N, 8) int32 — nearest E8int point.
    """
    x_f32 = x.float()

    u0 = (torch.round(x / 2) * 2).to(torch.int32)               # nearest even integer
    u0 = _fix_parity(u0, x_f32)

    u1 = (torch.round((x - 1) / 2) * 2 + 1).to(torch.int32)     # nearest odd integer
    u1 = _fix_parity(u1, x_f32)

    d0 = ((u0.float() - x_f32) ** 2).sum(dim=-1)
    d1 = ((u1.float() - x_f32) ** 2).sum(dim=-1)
    return torch.where(d0.unsqueeze(-1) <= d1.unsqueeze(-1), u0, u1)


def quantize_d4int(x: torch.Tensor) -> torch.Tensor:
    """
    Nearest-point decoder for D4int = 2·D4.

    D4int = { y ∈ Z^4 : all y_i even, Σ y_i ≡ 0 (mod 4) }.

    Unlike E8int, there is only one coset (all coords are always even), so no
    even/odd trial is needed — just round to the nearest even integer and fix
    the sum-parity constraint with _fix_parity.

    Args:
        x: (N, 4) bf16 — already-α-scaled input.
    Returns:
        (N, 4) int32 — nearest D4int point.
    """
    x_f32 = x.float()
    u = (torch.round(x / 2) * 2).to(torch.int32)    # nearest even integer
    return _fix_parity(u, x_f32)                      # enforce Σ u ≡ 0 (mod 4)


def quantize_a2int(x: torch.Tensor) -> torch.Tensor:
    """
    Nearest-point decoder for A2int = 2·A2 (hexagonal lattice).

    A2int = { (n_y·√3, n_x) : n_y, n_x ∈ Z, n_y + n_x ≡ 0 (mod 2) }.

    Physical coordinates: first column y_phys = n_y·√3, second x_phys = n_x.
    Stored representation: (n_y, n_x) as integers.

    Two cosets:
        Coset A: both n_y and n_x even  — nearest even rounding in each axis.
        Coset B: both n_y and n_x odd   — nearest odd rounding in each axis.

    SQRT3 appears exactly once (to convert y_phys → n_y float); the squared
    physical distance in stored coordinates is 3·(n_y − ny_f)² + (n_x − nx_f)².

    Args:
        x: (N, 2) bf16 — already-α-scaled input, columns [y_phys, x_phys].
    Returns:
        (N, 2) int32 — stored (n_y, n_x) of the nearest A2int point.
    """
    ny_f = x[:, 0].float() / SQRT3   # y_phys → float n_y (used for both cosets & distance)
    nx_f = x[:, 1].float()

    # Coset A — both even
    ny_a = (torch.round(ny_f / 2) * 2).to(torch.int32)
    nx_a = (torch.round(nx_f / 2) * 2).to(torch.int32)
    # Coset B — both odd
    ny_b = (torch.round((ny_f - 1) / 2) * 2 + 1).to(torch.int32)
    nx_b = (torch.round((nx_f - 1) / 2) * 2 + 1).to(torch.int32)

    # Squared physical distances: d²_phys = 3·(Δn_y)² + (Δn_x)²
    da2 = (ny_a.float() - ny_f) ** 2 * 3 + (nx_a.float() - nx_f) ** 2
    db2 = (ny_b.float() - ny_f) ** 2 * 3 + (nx_b.float() - nx_f) ** 2

    use_a = da2 <= db2
    return torch.stack([
        torch.where(use_a, ny_a, ny_b),
        torch.where(use_a, nx_a, nx_b),
    ], dim=1)


def quantize_z1(x: torch.Tensor) -> torch.Tensor:
    """
    Nearest-integer quantiser for Z1int.

    Each scalar is rounded independently to the nearest integer — no coset
    selection, no parity constraints.  The simplest possible lattice quantiser.

    Args:
        x: (N, 1) bf16 — already-α-scaled input.
    Returns:
        (N, 1) int32 — rounded integers.
    """
    return torch.round(x).to(torch.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
#
# Under high-resolution quantisation the noise is approximately uniform over
# the Voronoi cell, so for x ~ N(0, I_N):
#
#     SNR(α) = E‖α·x‖² / E‖α·x − Q(α·x)‖² ≈ α²·N / MSE_voronoi
#     ⇒  α   = √( RSNR_lin · MSE_analytic / N )
#
# Alpha is derived from the analytic MSE_voronoi (exact for each lattice),
# not the empirical estimate.  The empirical measurement at ALPHA_CAL = 50
# serves only as a sanity check: a deviation beyond tolerance indicates a
# bug in the quantiser or a mismatch in lattice scale.
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_alpha(rsnr_db: float, lattice: str = "e8int", n_cal: int = 1_000_000,
                    verbose: bool = False) -> float:
    """Return α achieving the requested SNR via the high-resolution formula.

    Dispatches to the correct quantiser and analytic MSE for the chosen lattice.
    Asserts that empirical Voronoi MSE is within MSE_VORONOI_TOL of the analytic value.
    A large deviation indicates a bug in the quantiser or a mismatch in lattice scale.
    """
    if lattice == "e8int":
        n_dim, mse_analytic, quantize_fn = N_DIM_E8, MSE_VORONOI_E8_ANALYTIC, quantize_e8int
        mse_label = "G_E8·N·256^(1/4)"
    elif lattice == "d4int":
        n_dim, mse_analytic, quantize_fn = N_DIM_D4, MSE_VORONOI_D4_ANALYTIC, quantize_d4int
        mse_label = "G_D4·N·√32"
    elif lattice == "a2int":
        n_dim, mse_analytic, quantize_fn = N_DIM_A2, MSE_VORONOI_A2_ANALYTIC, quantize_a2int
        mse_label = "G_A2·2·2√3 = 5/9"
    else:  # z1int
        n_dim, mse_analytic, quantize_fn = N_DIM_Z1, MSE_VORONOI_Z1_ANALYTIC, quantize_z1
        mse_label = "1/12"

    x_cal  = torch.randn(n_cal, n_dim, dtype=torch.bfloat16)
    ax_cal = (ALPHA_CAL * x_cal).to(torch.bfloat16)
    y_cal  = quantize_fn(ax_cal)

    # For A2int the stored n_y must be converted back to physical (×√3) before
    # computing the error; for all other lattices the stored values ARE the physical values.
    if lattice == "a2int":
        y_phys  = torch.stack([y_cal[:, 0].float() * SQRT3, y_cal[:, 1].float()], dim=1)
        err_cal = ax_cal.float() - y_phys
    else:
        err_cal = ax_cal.float() - y_cal.float()
    mse_voronoi = (err_cal ** 2).sum(dim=-1).mean().item()

    rel_err = abs(mse_voronoi - mse_analytic) / mse_analytic
    # Z1int Voronoi MSE (1/12 ≈ 0.083) is small enough that bfloat16 quantisation
    # noise (~0.013) adds ~15% on top of the analytic value — this is expected and
    # not a sign of a bug.  All other lattices have much larger Voronoi cells so
    # bf16 noise is negligible and the tighter 5% tolerance applies.
    tol = 0.20 if lattice == "z1int" else MSE_VORONOI_TOL
    assert rel_err < tol, (
        f"Empirical MSE_voronoi={mse_voronoi:.4f} deviates {rel_err*100:.1f}% "
        f"from analytic {mse_analytic:.4f} (tol {tol*100:.0f}%). "
        f"Check quantiser correctness and lattice scale."
    )

    alpha = math.sqrt(_rsnr_lin(rsnr_db) * mse_analytic / n_dim)
    if verbose:
        print(f"  MSE_voronoi empirical         = {mse_voronoi:.4f}")
        print(f"  MSE_voronoi analytic          = {mse_analytic:.4f}  ({mse_label})")
        print(f"  relative error                = {rel_err*100:.2f}%")
        print(f"  alpha                         = {alpha:.4f}")
    return alpha


# ─────────────────────────────────────────────────────────────────────────────
# Structural decomposition
#
# E8int / D4int share a two-step structure:
#   (A) Strip shared-parity redundancy → s_1..s_{N−1} + coset bit c (E8int only).
#   (B) Strip sum-parity redundancy → t from s_N.
# Encoder output: N symbols per vector encoded with a single Rice k:
#   s_1..s_{N−1}  (std ≈ α/2)  with Rice k
#   combined = 2·zz(t) + c     with Rice k
#     E8int: c ∈ {0,1} absorbs the coset bit into the LSB at zero extra cost.
#     D4int: c = 0 always; the LSB of combined is implicit (dropped in storage).
#
# A2int uses a different structure (parity strip only; two independent k values).
# Z1int has no structural decomposition at all.
# ─────────────────────────────────────────────────────────────────────────────

def structural_decompose(Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Strip the two deterministic redundancies of E8int.

     (A) All 8 coords share parity.  One coset bit c per vector encodes them all,
         then halve every coord:  s_i = (y_i − c) >> 1   (exact, y_i − c is even).
         Saves 7 LSBs/vector = 7/8 bit/scalar.

     (B) Σ y ≡ 0 (mod 4)  ⇒  Σ s ≡ 0 (mod 2)  ⇒  parity of s_8 is determined
         by s_1..s_7.  Halve s_8 once more:  t = (s_8 − known_parity) >> 1.
         Saves 1 bit/vector = 1/8 bit/scalar.

    Total = 8 deterministic bits/vector = exactly 1 bit/scalar.

    Returns:
        c   : (N,)    int32 in {0, 1} — coset (parity) bit per vector.
        S7  : (N, 7)  int32           — first 7 halved coords.
        T   : (N,)    int32           — s_8 with its forced parity halved out.
    """
    c        = (Y[:, 0] & 1).to(torch.int32)
    S        = (Y - c.unsqueeze(1)) >> 1
    known_p8 = S[:, :7].sum(dim=-1) & 1
    T        = (S[:, 7] - known_p8) >> 1
    return c, S[:, :7], T


def structural_decompose_d4(Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Strip the single deterministic redundancy of D4int.

    D4int = 2·D4, so all coords are always even.  After halving, the 4 coords
    satisfy Σ s_i ≡ 0 (mod 2), giving one forced parity bit:

        s_i = y_i >> 1              (exact since all y_i even)
        p   = (s_1 + s_2 + s_3) & 1    (forced parity of s_4)
        t   = (s_4 − p) >> 1        (strip the forced parity bit)

    There is no coset bit (c = 0 always for D4int).
    Savings: 1 bit/vector = 1/4 bit/scalar.

    Returns:
        S3 : (N, 3) int32 — first 3 halved coords.
        T  : (N,)   int32 — s_4 with its forced parity halved out.
    """
    S       = Y >> 1                              # s_i = y_i / 2 (exact, all y_i even)
    known_p = S[:, :3].sum(dim=-1) & 1            # forced parity of s_4
    T       = (S[:, 3] - known_p) >> 1            # strip the forced parity bit
    return S[:, :3], T


def structural_decompose_a2int(Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Strip the single deterministic redundancy of A2int.

    A2int stored as (n_y, n_x) with n_y + n_x ≡ 0 (mod 2), so the parity of
    n_y is fully determined by n_x.  Strip it:

        p   = n_x & 1           (forced parity of n_y)
        t_y = (n_y − p) >> 1   (strip the forced parity bit)

    Savings: 1 bit/vector = 1/2 bit/scalar.

    Returns:
        T_y : (N,) int32 — n_y with its forced parity bit stripped.
        N_x : (N,) int32 — n_x unchanged.
    """
    n_x = Y[:, 1]
    p   = n_x & 1               # forced parity of n_y (= parity of n_x)
    t_y = (Y[:, 0] - p) >> 1   # strip forced parity bit
    return t_y, n_x


# ─────────────────────────────────────────────────────────────────────────────
# Golomb-Rice coding
#
# For a non-negative integer z and parameter k, the code length is
#     (z >> k)  +  1  +  k    bits.
# (q = z >> k coded in unary as q+1 bits; remainder z mod 2^k in k bits.)
#
# Signed inputs are zig-zag mapped first:  n ≥ 0 → 2n,  n < 0 → -2n - 1.
#
# Why Rice and not arithmetic / tANS?  Stateless, table-free, trivially
# decoded.  The price is mild: Rice is exactly optimal for geometric (≈
# Laplacian) sources, and our residuals are Gaussian-shaped.  At 21 dB the
# Rice penalty over an ideal entropy coder is ≈ 0.12 bit/scalar — small
# enough that the simplicity is worth it.
# ─────────────────────────────────────────────────────────────────────────────

def _zigzag(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, 2 * x, -2 * x - 1)


def rice_bps(values: torch.Tensor, k: int) -> float:
    """Mean bits/value of Rice-k applied to a (signed) integer tensor."""
    zz = _zigzag(values)
    return ((zz >> k).float() + (1 + k)).mean().item()


def rice_optimal(values: torch.Tensor,
                 k_range: Iterable[int] = RICE_K_RANGE
                 ) -> tuple[int, float, list[float]]:
    """Return (best_k, best_bps, all_bps_in_order_of_k_range)."""
    ks       = list(k_range)
    bps_list = [rice_bps(values, k) for k in ks]
    i_best   = min(range(len(bps_list)), key=bps_list.__getitem__)
    return ks[i_best], bps_list[i_best], bps_list


# ─────────────────────────────────────────────────────────────────────────────
# One operating point
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(rsnr_db: float, n_vectors: int = 100_000, lattice: str = "e8int",
                   *, with_lz: bool = True, verbose_calibrate: bool = False) -> dict:
    """
    Run the full pipeline at one target SNR and report every metric.

    LZ measurements are emitted as NaN whenever max|Y| > 127 (raw int8 storage
    is unsafe in that regime).  Every other metric is always reported.
    """
    if lattice == "e8int":
        n_dim, g_lattice = N_DIM_E8, G_E8
    elif lattice == "d4int":
        n_dim, g_lattice = N_DIM_D4, G_D4
    elif lattice == "a2int":
        n_dim, g_lattice = N_DIM_A2, G_A2
    else:  # z1int
        n_dim, g_lattice = N_DIM_Z1, G_Z1

    alpha = calibrate_alpha(rsnr_db, lattice, verbose=verbose_calibrate)

    X  = torch.randn(n_vectors, n_dim, dtype=torch.bfloat16)
    aX = (alpha * X).to(torch.bfloat16)

    if lattice == "e8int":
        Y = quantize_e8int(aX)
    elif lattice == "d4int":
        Y = quantize_d4int(aX)
    elif lattice == "a2int":
        Y = quantize_a2int(aX)
    else:  # z1int
        Y = quantize_z1(aX)

    # For A2int, reconstruct physical values (n_y → n_y·√3) before computing SNR.
    # For all other lattices the stored values are the physical values.
    if lattice == "a2int":
        Y_phys   = torch.stack([Y[:, 0].float() * SQRT3, Y[:, 1].float()], dim=1)
        err_main = aX.float() - Y_phys
    else:
        err_main = aX.float() - Y.float()

    signal_power     = (aX.float() ** 2).sum(dim=-1).mean().item()
    noise_power      = (err_main  ** 2).sum(dim=-1).mean().item()
    snr_empirical_db = 10 * math.log10(signal_power / noise_power)

    # 8-bit budget: check the stored integer values (must fit in int8).
    p_overflow = (Y.abs() > 127).float().mean().item()
    y_abs_max  = int(Y.abs().max().item())

    # Marginal scalar entropy of the stored representation.
    scalars    = Y.reshape(-1)
    vals, cnts = torch.unique(scalars, return_counts=True)
    p_s        = cnts.float() / cnts.sum()
    H_scalar   = -(p_s * p_s.log2()).sum().item()
    n_unique   = int(vals.shape[0])

    # ── Structural decomposition + Rice coding ────────────────────────────────
    if lattice == "z1int":
        # Z1int: no structural decomposition.  Rice code each rounded scalar directly.
        rice_k, _, _  = rice_optimal(Y.reshape(-1))
        per_vec_bits  = (_zigzag(Y.reshape(-1)) >> rice_k).float() + (1 + rice_k)
        rice_k_bps    = per_vec_bits.mean().item()
        rice_k_y      = rice_k
        rice_k_x      = rice_k
        rice_k_bps_s  = rice_k_bps   # single symbol stream
        rice_k_bps_comb = 0.0

    elif lattice == "a2int":
        # A2int: strip parity bit from n_y; encode t_y and n_x independently.
        # std(t_y) ≈ α/(2√3) ≈ 0.29α,  std(n_x) ≈ α  → different k values.
        T_y, N_x = structural_decompose_a2int(Y)

        rice_k_y, rice_k_bps_s, _ = rice_optimal(T_y)
        rice_k_x, rice_k_bps_comb, _ = rice_optimal(N_x)

        bits_ty_per_vec = (_zigzag(T_y) >> rice_k_y).float() + (1 + rice_k_y)
        bits_nx_per_vec = (_zigzag(N_x) >> rice_k_x).float() + (1 + rice_k_x)
        per_vec_bits    = bits_ty_per_vec + bits_nx_per_vec
        rice_k_bps      = (rice_k_bps_s + rice_k_bps_comb) / n_dim
        rice_k          = ""          # no single k for A2int

    else:
        # E8int / D4int: single k for all symbols.
        n_s_syms = n_dim - 1   # s-symbols: 7 for E8int, 3 for D4int

        if lattice == "e8int":
            c, S_syms, T = structural_decompose(Y)
        else:
            S_syms, T = structural_decompose_d4(Y)

        # Verify k_s == k_t + 1 (std(s) ≈ 2·std(t) so optimal k shifts by 1).
        rice_k,     rice_k_bps_s, _ = rice_optimal(S_syms.reshape(-1))
        k_t_verify, _,             _ = rice_optimal(T)
        if rice_k != k_t_verify + 1:
            print(f"WARNING: k_s={rice_k} != k_t+1={k_t_verify+1} at {rsnr_db} dB "
                  f"({_lattice_display(lattice)}) — single-k assumption does not hold; "
                  f"combined coding is suboptimal for t.")

        # combined = 2·zz(T) + c:
        #   E8int: c ∈ {0,1} absorbed into the LSB of combined at zero extra cost.
        #   D4int: c = 0 always; the always-zero LSB of combined is implicit.
        #          Effective k for t: k_t = rice_k − 1.
        if lattice == "e8int":
            combined          = 2 * _zigzag(T) + c
            bits_comb_per_vec = (combined >> rice_k).float() + (1 + rice_k)
        else:
            k_t               = max(0, rice_k - 1)
            bits_comb_per_vec = (_zigzag(T) >> k_t).float() + (1 + k_t)

        rice_k_bps_comb = bits_comb_per_vec.mean().item()
        rice_k_y        = rice_k
        rice_k_x        = rice_k

        zz_s           = _zigzag(S_syms)
        bits_s_per_vec = ((zz_s >> rice_k).float() + (1 + rice_k)).sum(dim=-1)
        per_vec_bits   = bits_s_per_vec + bits_comb_per_vec
        rice_k_bps     = (n_s_syms * rice_k_bps_s + rice_k_bps_comb) / n_dim

    # ── 128-scalar block sizing ───────────────────────────────────────────────
    # Max: sum of the heaviest block_vecs vectors — worst-case burst for any aligned block.
    # Avg: mean per-vector cost × block_vecs — expected bits per 128-scalar block.
    block_vecs          = 128 // n_dim   # 16 (E8int), 32 (D4int), 64 (A2int), 128 (Z1int)
    max_128scalar_bits  = math.ceil(per_vec_bits.topk(block_vecs).values.sum().item())
    avg_128scalar_bits  = math.ceil(per_vec_bits.mean().item() * block_vecs)
    max_128scalar_bytes = math.ceil(max_128scalar_bits / 8)
    avg_128scalar_bytes = math.ceil(avg_128scalar_bits / 8)

    # ── Theoretical bounds ────────────────────────────────────────────────────
    rd_bound_bps      = 0.5 * math.log2(_rsnr_lin(rsnr_db))
    lattice_ideal_bps = rd_bound_bps + 0.5 * math.log2(2 * math.pi * math.e * g_lattice)

    out = {
        "lattice":             lattice,
        "n_dim":               n_dim,
        "snr_req_db":          rsnr_db,
        "alpha":               alpha,
        "snr_act_db":          snr_empirical_db,
        "P_8b_ovf":            p_overflow,
        "y_abs_max":           y_abs_max,
        "H_scalar":            H_scalar,
        "n_unique_vals":       n_unique,
        "rice_k":              rice_k,
        "rice_k_y":            rice_k_y,               # k for t_y (A2int) or same as rice_k
        "rice_k_x":            rice_k_x,               # k for n_x (A2int) or same as rice_k
        "rice_k_bps_s":        rice_k_bps_s,          # per s-symbol / t_y; kept for verbose print
        "rice_k_bps_comb":     rice_k_bps_comb,       # per combined / n_x; kept for verbose print
        "rice_bps":            rice_k_bps,
        "rd_bound_bps":        rd_bound_bps,
        "lattice_ideal_bps":   lattice_ideal_bps,
        "avg_128scalar_bytes": avg_128scalar_bytes,
        "max_128scalar_bytes": max_128scalar_bytes,
        "avg_128scalar_bits":  avg_128scalar_bits,
        "max_128scalar_bits":  max_128scalar_bits,
    }

    if with_lz:
        nan = float("nan")
        if y_abs_max <= 127:
            raw       = Y.to(torch.int8).numpy().tobytes()
            n_scalars = Y.numel()
            zlib_flat = zlib.compress(raw, level=9)
            lzma_flat = lzma.compress(raw, preset=9)
            zlib_cols = sum(
                len(zlib.compress(Y[:, i].to(torch.int8).numpy().tobytes(), level=9))
                for i in range(n_dim)
            )
            lzma_cols = sum(
                len(lzma.compress(Y[:, i].to(torch.int8).numpy().tobytes(), preset=9))
                for i in range(n_dim)
            )
            out.update({
                "zlib_flat_bps": len(zlib_flat) * 8 / n_scalars,
                "lzma_flat_bps": len(lzma_flat) * 8 / n_scalars,
                "zlib_cols_bps": zlib_cols * 8 / n_scalars,
                "lzma_cols_bps": lzma_cols * 8 / n_scalars,
            })
        else:
            out.update({
                "zlib_flat_bps": nan, "lzma_flat_bps": nan,
                "zlib_cols_bps": nan, "lzma_cols_bps": nan,
            })

    return out


# ─────────────────────────────────────────────────────────────────────────────

