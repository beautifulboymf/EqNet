"""Data: synthetic regression (for MLP) and MNIST (for CNN)."""
from __future__ import annotations

import numpy as np
import torch


# =============================================================================
# Synthetic regression for the MLP:
# y = sin(2 * w*·x) + 0.05 * ε,    x ∈ ℝ¹⁰ standard normal,
# w* a fixed unit vector. 2000 train / 500 test.
# =============================================================================

DATA_SEED = 12345
N_TRAIN = 2000
N_TEST = 500
D_IN = 10
NOISE_STD = 0.05
FREQ = 2.0


def synthetic_regression(device: str = "cpu") -> dict:
    g = np.random.default_rng(DATA_SEED)
    w_star = g.standard_normal(D_IN).astype(np.float32)
    w_star /= np.linalg.norm(w_star)
    x_tr = g.standard_normal((N_TRAIN, D_IN)).astype(np.float32)
    x_te = g.standard_normal((N_TEST, D_IN)).astype(np.float32)
    y_tr = np.sin(FREQ * x_tr @ w_star) + NOISE_STD * g.standard_normal(N_TRAIN).astype(np.float32)
    y_te = np.sin(FREQ * x_te @ w_star) + NOISE_STD * g.standard_normal(N_TEST).astype(np.float32)
    t = lambda a: torch.from_numpy(a).to(device)
    return {
        "w_star": w_star,
        "x_train": t(x_tr),
        "y_train": t(y_tr).unsqueeze(1),
        "x_test": t(x_te),
        "y_test": t(y_te).unsqueeze(1),
    }


# =============================================================================
# MNIST subset loader
# =============================================================================

def mnist_subset(n: int, data_dir: str, device: str, seed: int,
                   train: bool = True):
    from torchvision import datasets, transforms
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    ds = datasets.MNIST(data_dir, train=train, download=False, transform=tfm)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    xs, ys = [], []
    for i in idx:
        x, y = ds[int(i)]
        xs.append(x); ys.append(y)
    return torch.stack(xs).to(device), torch.tensor(ys).to(device)
