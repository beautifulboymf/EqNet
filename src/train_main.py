"""Train families of MLP and CNN networks for use in the experiments.

  python -m src.train_main mlp --n 50
  python -m src.train_main cnn --n 10 --data_dir mnist_data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training import train_mlp, train_cnn, save_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["mlp", "cnn"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--data_dir", default="mnist_data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.target == "mlp":
        out_dir = Path(args.out or "ckpts/mlp")
        seed_base = 1000
        train_fn = lambda s: train_mlp(seed=s, device=args.device)
    else:
        out_dir = Path(args.out or "ckpts/cnn")
        seed_base = 7000
        train_fn = lambda s: train_cnn(seed=s, data_dir=args.data_dir,
                                          epochs=args.epochs, device=args.device)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for i in range(args.n):
        seed = seed_base + i
        model, meta = train_fn(seed)
        path = out_dir / f"{args.target.upper()}_{i:03d}.pt"
        save_model(model, path, meta)
        print(f"[{args.target.upper()} {i+1}/{args.n}] seed={seed} "
              f"train={meta['train_loss']:.5f}", flush=True)
        summary.append(meta)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {len(summary)} ckpts → {out_dir}")


if __name__ == "__main__":
    main()
