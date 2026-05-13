"""Measure local output-preserving directions and run parameter walk tests.

This is the canonical script behind the README spectrum and walk-test result
files. It deliberately keeps the models small: the goal is to validate the
measurement logic, not to run a large benchmark.

Examples
--------
python -m src.run_local_geometry mlp \
    --ckpt_dir ckpts/mlp --anchors MLP_000 MLP_010 \
    --out_json results/walk_test_mlp.json

python -m src.run_local_geometry cnn \
    --ckpt_dir ckpts/cnn --anchors CNN_000 CNN_002 CNN_004 \
    --data_dir mnist_data --n_samples 1500 \
    --out_json results/walk_test_cnn.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import mnist_subset, synthetic_regression
from jacobian import compute_jacobian_cnn, compute_jacobian_mlp
from models import (
    CNN,
    MLP,
    cnn_params_to_vector,
    cnn_vector_to_state,
    gauge_basis_orthonormal,
    mlp_params_to_vector,
    mlp_vector_to_params,
)


def _load_model(target: str, ckpt: Path, device: str):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    if target == "mlp":
        model = MLP()
    else:
        model = CNN()
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model.to(device), blob.get("meta", {})


def _state_from_flat(target: str, model, flat: torch.Tensor):
    if target == "mlp":
        return mlp_vector_to_params(flat, model.d_in, model.d_hidden, model.d_out)
    return cnn_vector_to_state(flat, model.c1, model.c2, model.fc_out)


def _loss_at(target: str, model, flat_np: np.ndarray, x, y, device: str) -> float:
    flat = torch.from_numpy(flat_np.astype(np.float32)).to(device)
    state = _state_from_flat(target, model, flat)
    with torch.no_grad():
        out = functional_call(model, state, (x,))
        if target == "mlp":
            return float(((out - y) ** 2).mean().item())
        return float(F.cross_entropy(out, y).item())


def _orthonormal_rows(mat: np.ndarray, n_keep: int | None = None,
                      tol: float = 1e-7) -> np.ndarray:
    if mat.size == 0:
        return mat.reshape(0, mat.shape[-1])
    u, s, _ = np.linalg.svd(mat.T, full_matrices=False)
    keep = s > tol
    if n_keep is not None:
        idx = np.flatnonzero(keep)[:n_keep]
    else:
        idx = np.flatnonzero(keep)
    return u[:, idx].T


def _extra_basis(v_small: np.ndarray, gauge: np.ndarray,
                 extra_dim: int) -> np.ndarray:
    if extra_dim <= 0:
        return np.zeros((0, v_small.shape[0]), dtype=np.float64)
    # v_small has shape (P, k). Remove the analytical gauge component first.
    resid = v_small - gauge.T @ (gauge @ v_small)
    return _orthonormal_rows(resid.T, n_keep=extra_dim)


def _sample_losses(flat_np: np.ndarray, basis: np.ndarray, eps_grid: list[float],
                   n_walks: int, loss_fn, rng: np.random.Generator) -> dict:
    if basis.shape[0] == 0:
        return {"available": False, "eps": eps_grid, "losses": []}
    losses = []
    for eps in eps_grid:
        row = []
        for _ in range(n_walks):
            coeff = rng.standard_normal(basis.shape[0])
            coeff /= max(np.linalg.norm(coeff), 1e-30)
            v = coeff @ basis
            v /= max(np.linalg.norm(v), 1e-30)
            row.append(loss_fn(flat_np + eps * v))
        losses.append(row)
    return {"available": True, "eps": eps_grid, "losses": losses}


def measure_anchor(target: str, ckpt: Path, args, rng: np.random.Generator) -> dict:
    model, meta = _load_model(target, ckpt, args.device)
    if target == "mlp":
        data = synthetic_regression(args.device)
        x, y = data["x_train"], data["y_train"]
        flat = mlp_params_to_vector(model).to(args.device)
        jac = compute_jacobian_mlp(model, x, flat).double()
    else:
        x, y = mnist_subset(args.n_samples, args.data_dir, args.device, args.seed)
        flat = cnn_params_to_vector(model).to(args.device)
        jac = compute_jacobian_cnn(model, x, flat,
                                   chunk_size=args.chunk_size).double()

    flat_np = flat.detach().cpu().double().numpy()
    gauge = gauge_basis_orthonormal(model).detach().cpu().double().numpy()
    gauge_dim = int(gauge.shape[0])

    j_cpu = jac.cpu().double().numpy()
    jg_norms = np.linalg.norm(j_cpu @ gauge.T, axis=0)
    tau = max(5.0 * float(jg_norms.max()), args.min_tau)

    m = j_cpu.T @ j_cpu
    m = 0.5 * (m + m.T)
    eigvals, eigvecs = np.linalg.eigh(m)
    sigmas = np.sqrt(np.clip(eigvals, 0.0, None))
    small = sigmas < tau
    tangent_dim = int(small.sum())
    extra_dim = max(tangent_dim - gauge_dim, 0)

    v_small = eigvecs[:, small]
    extras = _extra_basis(v_small, gauge, extra_dim)
    top = eigvecs[:, -min(args.n_top, eigvecs.shape[1]):].T
    bottom = eigvecs[:, :min(args.n_bottom, eigvecs.shape[1])].T

    loss_fn = lambda p: _loss_at(target, model, p, x, y, args.device)
    l0 = loss_fn(flat_np)
    p = flat_np.size
    random_basis = _orthonormal_rows(rng.standard_normal((args.n_walks, p)))

    walk_results = {
        "gauge": _sample_losses(flat_np, gauge, args.eps_grid, args.n_walks,
                                loss_fn, rng),
        "extra": _sample_losses(flat_np, extras, args.eps_grid, args.n_walks,
                                loss_fn, rng),
        "curved": _sample_losses(flat_np, top, args.eps_grid, args.n_walks,
                                 loss_fn, rng),
        "random": _sample_losses(flat_np, random_basis, args.eps_grid,
                                  args.n_walks, loss_fn, rng),
    }

    cos = np.linalg.svd(gauge @ v_small, compute_uv=False) if v_small.size else []
    record = {
        "ckpt": str(ckpt),
        "meta": meta,
        "L0": l0,
        "P": int(p),
        "gauge_dim": gauge_dim,
        "tangent_dim": tangent_dim,
        "extra_dim": extra_dim,
        "tau": float(tau),
        "JG_norm_max": float(jg_norms.max()),
        "JG_norm_min": float(jg_norms.min()),
        "principal_angle_cos_all": [float(v) for v in cos],
        "walk_eps_grid": args.eps_grid,
        "walk_results": walk_results,
    }

    args.out_npz_dir.mkdir(parents=True, exist_ok=True)
    prefix = "tangent_mlp" if target == "mlp" else "tangent_cnn"
    npz_path = args.out_npz_dir / f"{prefix}_{ckpt.stem}.npz"
    np.savez(
        npz_path,
        S=sigmas,
        gauge_basis=gauge,
        extra_basis=extras,
        Vh_top10=top[:10],
        Vh_bottom30=bottom[:30],
    )
    record["npz"] = str(npz_path)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["mlp", "cnn"])
    parser.add_argument("--anchors", nargs="+", default=None)
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--data_dir", default="mnist_data")
    parser.add_argument("--n_samples", type=int, default=1500)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--eps_grid", nargs="+", type=float,
                        default=[0.0, 0.01, 0.05, 0.1, 0.3, 1.0])
    parser.add_argument("--n_walks", type=int, default=3)
    parser.add_argument("--n_top", type=int, default=20)
    parser.add_argument("--n_bottom", type=int, default=30)
    parser.add_argument("--min_tau", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--out_npz_dir", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.ckpt_dir is None:
        args.ckpt_dir = "ckpts/mlp" if args.target == "mlp" else "ckpts/cnn"
    if args.anchors is None:
        args.anchors = ["MLP_000"] if args.target == "mlp" else ["CNN_000"]
    if args.out_json is None:
        args.out_json = f"results/walk_test_{args.target}.json"
    args.out_npz_dir = Path(args.out_npz_dir)

    rng = np.random.default_rng(args.seed)
    records = []
    for name in args.anchors:
        ckpt = Path(args.ckpt_dir) / f"{name}.pt"
        if not ckpt.exists():
            print(f"[skip] missing checkpoint: {ckpt}", flush=True)
            continue
        print(f"[measure] {ckpt}", flush=True)
        records.append(measure_anchor(args.target, ckpt, args, rng))
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(records, f, indent=2)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
