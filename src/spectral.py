"""Spectral projector Π(θ) onto the small-eigenvalue subspace of M(θ) = J^T J.

  Π(θ) = ∮_{|z|<r} (zI − M(θ))⁻¹ dz / (2πi)
       = Σ_{i : λ_i < r} v_i v_iᵀ

is basis-ordering-independent and analytic in θ wherever the spectral gap is
preserved. The first-order variation is off-diagonal in the spectral basis
(Sylvester form), and is what the closed-loop / FD-bracket tests pick up.

Functions
---------
spectral_projector_from_M(M, r, *, return_eig=False, device='cpu')
delta_Pi_sylvester(eigvals, eigvecs, dM, r)
gap_diag(eigvals, r)
"""
from __future__ import annotations

import numpy as np
import torch


def spectral_projector_from_M(M: np.ndarray, r: float,
                                return_eig: bool = False,
                                device: str = "cpu"):
    """Return Π built from the eigvecs of M with eigval < r.

    For P > 1500, prefer device='cuda' (torch eigh on GPU)."""
    M_sym = 0.5 * (M + M.T)
    if device == "cuda" and M_sym.shape[0] > 1500:
        Mt = torch.from_numpy(M_sym.astype(np.float64)).to(device)
        eigvals_t, eigvecs_t = torch.linalg.eigh(Mt)
        eigvals = eigvals_t.detach().cpu().numpy()
        eigvecs = eigvecs_t.detach().cpu().numpy()
    else:
        eigvals, eigvecs = np.linalg.eigh(M_sym)
    small = eigvals < r
    V = eigvecs[:, small]
    Pi = V @ V.T
    if return_eig:
        return Pi, eigvals, eigvecs
    return Pi


def gap_diag(eigvals: np.ndarray, r: float) -> dict:
    small = eigvals[eigvals < r]
    large = eigvals[eigvals >= r]
    if len(small) == 0 or len(large) == 0:
        return {"r": float(r), "clean": False, "reason": "no small/large"}
    return {
        "r": float(r),
        "n_small": int(len(small)),
        "n_large": int(len(large)),
        "lam_small_max": float(small.max()),
        "lam_large_min": float(large.min()),
        "gap_ratio": float(large.min() / max(small.max(), 1e-30)),
        "clean": bool(large.min() / max(small.max(), 1e-30) > 10.0),
    }


def delta_Pi_sylvester(eigvals: np.ndarray, eigvecs: np.ndarray,
                        dM: np.ndarray, r: float,
                        clip: float = 1e-8) -> np.ndarray:
    """First-order δΠ from δM in the eigenbasis of M(θ). The standard formula
    is δΠ_{ij} = δM_{ij} / (λ_i − λ_j) for i,j in different blocks (one small,
    one large), zero on diagonal blocks."""
    small = eigvals < r
    cross = np.logical_xor(small[:, None], small[None, :])
    dM_eig = eigvecs.T @ dM @ eigvecs
    denom = eigvals[:, None] - eigvals[None, :]
    safe = np.where(np.abs(denom) > clip,
                    denom,
                    np.sign(denom) * clip + (denom == 0) * clip)
    delta_eig = np.where(cross, dM_eig / safe, 0.0)
    delta_Pi = eigvecs @ delta_eig @ eigvecs.T
    return 0.5 * (delta_Pi + delta_Pi.T)
