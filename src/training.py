"""Train the MLP on the synthetic regression task, and the CNN on MNIST."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import MLP, CNN
from data import synthetic_regression


# =============================================================================
# MLP training (synthetic regression)
# =============================================================================

def train_mlp(seed: int, steps: int = 10000, lr: float = 1e-3, bs: int = 256,
                device: str = "cuda") -> tuple[MLP, dict]:
    torch.manual_seed(seed)
    model = MLP().to(device)
    data = synthetic_regression(device)
    x, y = data["x_train"], data["y_train"]
    n = x.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator(device="cpu").manual_seed(42)  # fixed dataloader seed
    for step in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g).to(device)
        loss = ((model(x[idx]) - y[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        train_loss = float(((model(x) - y) ** 2).mean().item())
        test_loss = float(((model(data["x_test"]) - data["y_test"]) ** 2).mean().item())
    return model, {"seed": seed, "train_loss": train_loss, "test_loss": test_loss}


def save_model(model, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                 "meta": meta}, path)


# =============================================================================
# CNN training (MNIST)
# =============================================================================

def _mnist_loaders(data_dir: str, bs: int):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)
    return (DataLoader(train, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True),
            DataLoader(test, batch_size=1024, shuffle=False, num_workers=2, pin_memory=True))


def train_cnn(seed: int, data_dir: str, epochs: int = 5, lr: float = 1e-3,
               bs: int = 128, device: str = "cuda") -> tuple[CNN, dict]:
    torch.manual_seed(seed)
    model = CNN().to(device)
    train_loader, test_loader = _mnist_loaders(data_dir, bs)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    def _eval(loader):
        tl, c, t = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                tl += crit(out, yb).item() * xb.size(0)
                c += (out.argmax(1) == yb).sum().item()
                t += xb.size(0)
        return tl / t, c / t
    tr_l, tr_a = _eval(train_loader)
    te_l, te_a = _eval(test_loader)
    return model, {"seed": seed, "train_loss": tr_l, "test_loss": te_l,
                     "train_acc": tr_a, "test_acc": te_a}
