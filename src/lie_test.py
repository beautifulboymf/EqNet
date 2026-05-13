"""Closed-loop Lie test and direct-FD Lie bracket on Π-projected vector fields.

Both end up measuring (∇_v Π) X₀, which by perturbation theory of Π satisfies
(∇_v Π) X₀ ∈ (I − Π). Documenting the result is half the contribution; the
other half is identifying the structural confound.

Public API
----------
closed_loop(p0, X, Y, eps, compute_M, r, eigh_device='cpu')
direct_bracket(p0, X, Y, compute_M, r, h, eigh_device='cpu')
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from spectral import spectral_projector_from_M


def closed_loop(p0: np.ndarray, X: np.ndarray, Y: np.ndarray, eps: float,
                compute_M: Callable[[np.ndarray], np.ndarray], r: float,
                eigh_device: str = "cpu") -> dict:
    """Walk θ → +X → +Y(transp) → −X(transp) → −Y(transp), each transport via
    Π(θ') applied to the original θ₀-vector (no renormalisation).
    Return endpoint displacement decomposition relative to Π(θ₀)."""
    Pi0 = spectral_projector_from_M(compute_M(p0), r, device=eigh_device)

    p1 = p0 + eps * X
    Pi1 = spectral_projector_from_M(compute_M(p1), r, device=eigh_device)
    Y1 = Pi1 @ Y

    p2 = p1 + eps * Y1
    Pi2 = spectral_projector_from_M(compute_M(p2), r, device=eigh_device)
    X2 = Pi2 @ X

    p3 = p2 - eps * X2
    Pi3 = spectral_projector_from_M(compute_M(p3), r, device=eigh_device)
    Y3 = Pi3 @ Y

    p4 = p3 - eps * Y3
    disp = p4 - p0
    disp_in = Pi0 @ disp
    disp_norm = float(np.linalg.norm(disp))
    in_norm = float(np.linalg.norm(disp_in))
    out_norm = float(np.linalg.norm(disp - disp_in))
    return {
        "eps": float(eps),
        "disp_norm": disp_norm,
        "in_norm": in_norm,
        "out_norm": out_norm,
        "in_frac": in_norm / max(disp_norm, 1e-30),
        "Pi0_trace": float(np.trace(Pi0)),
    }


def direct_bracket(p0: np.ndarray, X: np.ndarray, Y: np.ndarray,
                    compute_M: Callable[[np.ndarray], np.ndarray], r: float,
                    h: float, eigh_device: str = "cpu") -> dict:
    """Lie bracket of vector fields A(θ)=Π(θ)X, B(θ)=Π(θ)Y via central FD:

      [A, B](θ₀) = (∇_A B)(θ₀) − (∇_B A)(θ₀)

    where (∇_v F)(θ₀) ≈ [F(θ₀+hv) − F(θ₀−hv)] / 2h.
    Returns magnitude and Π(θ₀)-fraction of the result.
    """
    def A_field(theta):
        return spectral_projector_from_M(compute_M(theta), r, device=eigh_device) @ X

    def B_field(theta):
        return spectral_projector_from_M(compute_M(theta), r, device=eigh_device) @ Y

    A0 = A_field(p0)
    B0 = B_field(p0)
    grad_A_B = (B_field(p0 + h * A0) - B_field(p0 - h * A0)) / (2.0 * h)
    grad_B_A = (A_field(p0 + h * B0) - A_field(p0 - h * B0)) / (2.0 * h)
    bracket = grad_A_B - grad_B_A
    Pi0 = spectral_projector_from_M(compute_M(p0), r, device=eigh_device)
    in_disp = Pi0 @ bracket
    bn = float(np.linalg.norm(bracket))
    inn = float(np.linalg.norm(in_disp))
    outn = float(np.linalg.norm(bracket - in_disp))
    return {
        "h": float(h),
        "bracket_norm": bn,
        "in_norm": inn,
        "out_norm": outn,
        "in_frac": inn / max(bn, 1e-30),
    }
