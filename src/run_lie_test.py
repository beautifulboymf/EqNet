"""Experiment 2 — closed-loop and direct-bracket Lie tests on Π(θ).

Two settings:
  * --target mlp_gauge  : sanity baseline; samples come from the analytical
                          gauge basis (a closed abelian Lie sub-algebra).
  * --target cnn        : main experiment; samples come from Π(θ₀) of the CNN.

For each (anchor, pair) and each ε / h, run closed_loop and direct_bracket and
record |disp|, in/out_norm, and the in_frac. Output JSON for later plotting.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import (MLP, CNN,
                    mlp_params_to_vector, cnn_params_to_vector,
                    mlp_vector_to_params, cnn_vector_to_state,
                    gauge_basis_orthonormal)
from data import synthetic_regression, mnist_subset
from jacobian import compute_jacobian_mlp, compute_jacobian_cnn
from spectral import spectral_projector_from_M, gap_diag
from lie_test import closed_loop, direct_bracket


def _load(target: str, ckpt: Path, device: str):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    if target == "mlp_gauge":
        m = MLP(); m.load_state_dict(blob["state_dict"]); return m.to(device)
    m = CNN(); m.load_state_dict(blob["state_dict"]); return m.to(device)


def _make_M_fn(target: str, x: torch.Tensor, device: str, n_chunk: int = 100):
    if target == "mlp_gauge":
        def compute_M(flat_np):
            ft = torch.from_numpy(flat_np.astype(np.float32)).to(device)
            sd = mlp_vector_to_params(ft, d_in=10, d_hidden=64, d_out=1)
            m = MLP().to(device)
            m.load_state_dict({k: v.detach() for k, v in sd.items()})
            J = compute_jacobian_mlp(m, x, ft).double()
            return (J.T @ J).cpu().double().numpy()
    else:
        def compute_M(flat_np):
            ft = torch.from_numpy(flat_np.astype(np.float32)).to(device)
            sd = cnn_vector_to_state(ft, 8, 16, 10)
            m = CNN().to(device)
            m.load_state_dict({k: v.detach() for k, v in sd.items()})
            flat_for_jac = cnn_params_to_vector(m).to(device)
            J = compute_jacobian_cnn(m, x, flat_for_jac, chunk_size=n_chunk).double()
            return (J.T @ J).cpu().double().numpy()
    return compute_M


def _pick_r_from_eigvals(eigvals: np.ndarray, k_target: int) -> float:
    """Place r in the biggest gap log-spread around index k_target."""
    sv = np.sort(eigvals)
    # restrict to the neighbourhood of k_target
    lo = max(k_target - 5, 1)
    hi = min(k_target + 50, len(sv) - 1)
    ratios = sv[lo + 1: hi + 1] / np.maximum(sv[lo:hi], 1e-30)
    gi = int(np.argmax(ratios)) + lo
    return float(np.sqrt(max(sv[gi], 1e-30) * max(sv[gi + 1], 1e-30)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["mlp_gauge", "cnn"], required=True)
    ap.add_argument("--anchors", nargs="+", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--data_dir", default="mnist_data")
    ap.add_argument("--n_pairs", type=int, default=10)
    ap.add_argument("--n_samples", type=int, default=1500)
    ap.add_argument("--eps_grid", nargs="+", type=float,
                    default=[1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--h_grid", nargs="+", type=float, default=[1e-3, 1e-2])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2026)
    if args.target == "mlp_gauge":
        x = synthetic_regression(args.device)["x_train"]
    else:
        x, _ = mnist_subset(args.n_samples, args.data_dir, args.device, 20260512)
    compute_M = _make_M_fn(args.target, x, args.device)

    summary = []
    for name in args.anchors:
        ckpt = Path(args.ckpt_dir) / f"{name}.pt"
        if not ckpt.exists():
            print(f"[skip] {ckpt}"); continue
        print(f"\n=== {name} ===", flush=True)
        model = _load(args.target, ckpt, args.device)
        flat0 = (mlp_params_to_vector(model) if args.target == "mlp_gauge"
                  else cnn_params_to_vector(model)).detach().cpu().double().numpy()
        M0 = compute_M(flat0)
        eigvals0 = np.linalg.eigvalsh(0.5 * (M0 + M0.T))
        k_target = 64 if args.target == "mlp_gauge" else 90
        r = _pick_r_from_eigvals(eigvals0, k_target)
        diag = gap_diag(np.sort(eigvals0), r)
        print(f"  r = {r:.3e}; spectral gap n_small={diag['n_small']}, "
              f"n_large={diag['n_large']}, gap_ratio={diag['gap_ratio']:.3e}",
              flush=True)
        Pi0 = spectral_projector_from_M(M0, r, device=args.device)

        # Build pair sampler
        if args.target == "mlp_gauge":
            G = gauge_basis_orthonormal(model).detach().cpu().double().numpy()
            sampler_basis = G
        else:
            sampler_basis = Pi0  # sample from Π(θ₀)

        records = {"closed_loop": [], "direct_bracket": []}
        for pair_idx in range(args.n_pairs):
            if args.target == "mlp_gauge":
                cx = rng.standard_normal(sampler_basis.shape[0]); cx /= np.linalg.norm(cx)
                cy = rng.standard_normal(sampler_basis.shape[0]); cy /= np.linalg.norm(cy)
                X0 = sampler_basis.T @ cx; Y0 = sampler_basis.T @ cy
            else:
                a = rng.standard_normal(flat0.size)
                X0 = sampler_basis @ a
                b = rng.standard_normal(flat0.size)
                Y0 = sampler_basis @ b
                nx = np.linalg.norm(X0)
                if nx < 1e-8: continue
                X0 = X0 / nx
            Y0 = Y0 - (X0 @ Y0) * X0
            ny = np.linalg.norm(Y0)
            if ny < 1e-8: continue
            Y0 = Y0 / ny

            for eps in args.eps_grid:
                t0 = time.time()
                r_l = closed_loop(flat0, X0, Y0, eps, compute_M, r,
                                    eigh_device=args.device)
                r_l["pair_idx"] = pair_idx
                r_l["elapsed"] = time.time() - t0
                records["closed_loop"].append(r_l)
                print(f"  [loop]    pair {pair_idx} eps={eps:.0e} "
                      f"|disp|={r_l['disp_norm']:.3e}  in_frac={r_l['in_frac']:.3f}",
                      flush=True)

            for h in args.h_grid:
                t0 = time.time()
                r_b = direct_bracket(flat0, X0, Y0, compute_M, r, h,
                                       eigh_device=args.device)
                r_b["pair_idx"] = pair_idx
                r_b["elapsed"] = time.time() - t0
                records["direct_bracket"].append(r_b)
                print(f"  [bracket] pair {pair_idx} h={h:.0e} "
                      f"|[X,Y]|={r_b['bracket_norm']:.3e}  in_frac={r_b['in_frac']:.3f}",
                      flush=True)

        summary.append({"anchor": name, "r": r, "spectral_gap": diag,
                         "Pi0_trace": float(np.trace(Pi0)),
                         "records": records})
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
