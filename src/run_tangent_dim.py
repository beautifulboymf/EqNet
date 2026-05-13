"""Experiment 1 — tangent dimension vs Jacobian sample size on the CNN.

For each CNN anchor, sweep n_samples ∈ {500, 1000, 2500, 5000, 10000} and
measure the dimension of the small-eigenvalue subspace of M(θ) = J^T J. The
plateau at large n (well above the analytical gauge dim of 24) is the v3-
finding that extras are real, not a training-sample artifact.
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

from models import CNN, cnn_params_to_vector, gauge_basis_orthonormal
from data import mnist_subset
from jacobian import compute_jacobian_cnn


def load_cnn(ckpt_path: Path, device: str) -> CNN:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = CNN(); m.load_state_dict(blob["state_dict"])
    return m.to(device)


def measure_one(ckpt: Path, n: int, device: str, data_dir: str, seed: int):
    model = load_cnn(ckpt, device)
    flat = cnn_params_to_vector(model).to(device)
    X, _ = mnist_subset(n, data_dir, device, seed)
    G = gauge_basis_orthonormal(model).detach().cpu().double().numpy()
    kG = G.shape[0]

    t0 = time.time()
    J = compute_jacobian_cnn(model, X, flat, chunk_size=100).double()
    t_J = time.time() - t0

    JG = J @ torch.from_numpy(G.astype(np.float64)).T
    Jv_norms = torch.norm(JG, dim=0).numpy()

    t0 = time.time()
    J_gpu = J.to(device)
    M = J_gpu.T @ J_gpu
    M = 0.5 * (M + M.T)
    eigvals = torch.linalg.eigvalsh(M).cpu().numpy()
    t_eig = time.time() - t0

    tau = max(5.0 * float(Jv_norms.max()), 1e-7) ** 2  # threshold on eigval ~ σ^2
    sigmas = np.sqrt(np.clip(eigvals, 0.0, None))
    tau_sig = max(5.0 * float(Jv_norms.max()), 1e-7)

    small = sigmas < tau_sig
    n_small = int(small.sum())
    extra = max(n_small - kG, 0)
    return {
        "n_samples": n,
        "gauge_dim": int(kG),
        "tangent_dim": n_small,
        "extra_dim": extra,
        "tau": tau_sig,
        "jacobian_seconds": t_J,
        "eigh_seconds": t_eig,
        "sigma_min": float(sigmas[0]),
        "sigma_max": float(sigmas[-1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", nargs="+",
                    default=[f"CNN_{i:03d}" for i in [0, 2, 4, 6, 8]])
    ap.add_argument("--ckpt_dir", default="ckpts/cnn")
    ap.add_argument("--data_dir", default="mnist_data")
    ap.add_argument("--n_grid", nargs="+", type=int,
                    default=[500, 1000, 2500, 5000, 10000])
    ap.add_argument("--out", default="results/tangent_dim_scaling.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in args.anchors:
        ckpt = Path(args.ckpt_dir) / f"{name}.pt"
        if not ckpt.exists():
            print(f"[skip] {ckpt}"); continue
        print(f"\n=== {name} ===", flush=True)
        records = []
        for n in args.n_grid:
            print(f"  n_samples = {n}", flush=True)
            try:
                rec = measure_one(ckpt, n, args.device, args.data_dir, 20260512 + n)
            except Exception as e:
                rec = {"n_samples": n, "error": str(e)}
            print(f"    tangent={rec.get('tangent_dim','?')}  "
                  f"extra={rec.get('extra_dim','?')}", flush=True)
            records.append(rec)
        summary.append({"anchor": name, "records": records})
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
