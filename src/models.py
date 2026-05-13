"""Models and their analytical structural-gauge tangent bases.

Two architectures, each with the same shape of analytical gauge:
per-channel positive scaling on each ReLU layer.

  - MLP   :  ℝ¹⁰ → ReLU(64) → 1.       gauge dim = 64.
  - CNN   :  1×28×28 → conv(8)+ReLU+maxpool → conv(16)+ReLU+maxpool → fc(10).
            gauge dim = 8 + 16 = 24.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# MLP
# =============================================================================

class MLP(nn.Module):
    def __init__(self, d_in: int = 10, d_hidden: int = 64, d_out: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.d_in, self.d_hidden, self.d_out = d_in, d_hidden, d_out

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def mlp_params_to_vector(model: MLP) -> torch.Tensor:
    return torch.cat([
        model.fc1.weight.detach().reshape(-1),
        model.fc1.bias.detach().reshape(-1),
        model.fc2.weight.detach().reshape(-1),
        model.fc2.bias.detach().reshape(-1),
    ])


def mlp_vector_to_params(vec: torch.Tensor, d_in: int = 10,
                           d_hidden: int = 64, d_out: int = 1):
    i = 0
    w1 = vec[i:i + d_hidden * d_in].reshape(d_hidden, d_in); i += d_hidden * d_in
    b1 = vec[i:i + d_hidden].reshape(d_hidden);                  i += d_hidden
    w2 = vec[i:i + d_out * d_hidden].reshape(d_out, d_hidden); i += d_out * d_hidden
    b2 = vec[i:i + d_out].reshape(d_out);                       i += d_out
    return {"fc1.weight": w1, "fc1.bias": b1,
             "fc2.weight": w2, "fc2.bias": b2}


def mlp_gauge_generators(model: MLP) -> torch.Tensor:
    """Per-hidden-neuron scaling generator in flat-parameter coordinates.

    For neuron i, the infinitesimal scaling at s = 1 sends
        δ W1[i, :]   = + W1[i, :]
        δ b1[i]      = + b1[i]
        δ W2[:, i]   = - W2[:, i]

    Returns shape (d_hidden, P).
    """
    d_in, H, d_out = model.d_in, model.d_hidden, model.d_out
    P = H * d_in + H + d_out * H + d_out
    device = next(model.parameters()).device
    W1 = model.fc1.weight.detach()
    b1 = model.fc1.bias.detach()
    W2 = model.fc2.weight.detach()
    G = torch.zeros(H, P, device=device, dtype=W1.dtype)
    s_w1, s_b1, s_w2 = 0, H * d_in, H * d_in + H
    for i in range(H):
        G[i, s_w1 + i * d_in : s_w1 + (i + 1) * d_in] = W1[i, :]
        G[i, s_b1 + i] = b1[i]
        for k in range(d_out):
            G[i, s_w2 + k * H + i] = -W2[k, i]
    return G


# =============================================================================
# CNN
# =============================================================================

class CNN(nn.Module):
    def __init__(self, c1: int = 8, c2: int = 16, fc_out: int = 10):
        super().__init__()
        self.c1, self.c2, self.fc_out = c1, c2, fc_out
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=True)
        self.fc = nn.Linear(c2 * 7 * 7, fc_out, bias=True)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc(x.flatten(1))


def cnn_n_params(model: CNN) -> int:
    return sum(p.numel() for p in model.parameters())


def cnn_params_to_vector(model: CNN) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for _, p in model.named_parameters()])


def cnn_vector_to_state(vec: torch.Tensor, c1: int = 8, c2: int = 16,
                          fc_out: int = 10):
    layout = _cnn_layout(c1, c2, fc_out)
    sd = {}
    for name, shape in layout["shapes"].items():
        s, e = layout["slices"][name]
        sd[name] = vec[s:e].reshape(shape)
    return sd


def _cnn_layout(c1: int = 8, c2: int = 16, fc_out: int = 10):
    shapes = {
        "conv1.weight": (c1, 1, 3, 3),
        "conv1.bias":   (c1,),
        "conv2.weight": (c2, c1, 3, 3),
        "conv2.bias":   (c2,),
        "fc.weight":    (fc_out, c2 * 7 * 7),
        "fc.bias":      (fc_out,),
    }
    slices = {}
    cur = 0
    for k, sh in shapes.items():
        n = int(np.prod(sh))
        slices[k] = (cur, cur + n)
        cur += n
    return {"shapes": shapes, "slices": slices, "total": cur}


def cnn_gauge_generators(model: CNN) -> torch.Tensor:
    """Per-channel scaling generators for the CNN's two ReLU layers.

    For each conv1 channel i (i = 0 .. c1−1):
        δ conv1.weight[i, :, :, :]      = + conv1.weight[i, :, :, :]
        δ conv1.bias[i]                 = + conv1.bias[i]
        δ conv2.weight[:, i, :, :]      = − conv2.weight[:, i, :, :]

    For each conv2 channel j (j = 0 .. c2−1):
        δ conv2.weight[j, :, :, :]      = + conv2.weight[j, :, :, :]
        δ conv2.bias[j]                 = + conv2.bias[j]
        δ fc.weight[:, j·49 : (j+1)·49] = − fc.weight[:, j·49 : (j+1)·49]

    Returns shape (c1 + c2, P).
    """
    c1, c2, fc_out = model.c1, model.c2, model.fc_out
    layout = _cnn_layout(c1, c2, fc_out)
    P = layout["total"]
    device = next(model.parameters()).device
    W1, b1 = model.conv1.weight.detach(), model.conv1.bias.detach()
    W2, b2 = model.conv2.weight.detach(), model.conv2.bias.detach()
    Wf = model.fc.weight.detach()
    G = torch.zeros(c1 + c2, P, device=device, dtype=W1.dtype)
    s_w1, _ = layout["slices"]["conv1.weight"]
    s_b1, _ = layout["slices"]["conv1.bias"]
    s_w2, _ = layout["slices"]["conv2.weight"]
    s_b2, _ = layout["slices"]["conv2.bias"]
    s_wf, _ = layout["slices"]["fc.weight"]
    per_chan_1 = 1 * 3 * 3
    per_chan_2_block = c1 * 3 * 3
    fc_cols_per_chan = 49
    # conv1 generators
    for i in range(c1):
        G[i, s_w1 + i * per_chan_1 : s_w1 + (i + 1) * per_chan_1] = W1[i].reshape(-1)
        G[i, s_b1 + i] = b1[i]
        for j in range(c2):
            base = s_w2 + j * per_chan_2_block + i * 9
            G[i, base : base + 9] = -W2[j, i, :, :].reshape(-1)
    # conv2 generators
    for j in range(c2):
        row = c1 + j
        G[row, s_w2 + j * per_chan_2_block : s_w2 + (j + 1) * per_chan_2_block] = W2[j].reshape(-1)
        G[row, s_b2 + j] = b2[j]
        for k in range(fc_out):
            base = s_wf + k * (c2 * fc_cols_per_chan) + j * fc_cols_per_chan
            G[row, base : base + fc_cols_per_chan] = -Wf[k, j * fc_cols_per_chan : (j + 1) * fc_cols_per_chan]
    return G


def gauge_basis_orthonormal(model) -> torch.Tensor:
    """Return a row-orthonormal version of the analytical gauge generators.

    Drops generators with norm < 1e-10 (dead channels)."""
    if isinstance(model, MLP):
        G = mlp_gauge_generators(model)
    elif isinstance(model, CNN):
        G = cnn_gauge_generators(model)
    else:
        raise TypeError(type(model))
    norms = torch.norm(G, dim=1)
    keep = norms > 1e-10
    if keep.sum() == 0:
        return G[:0]
    Gk = G[keep]
    Q, _ = torch.linalg.qr(Gk.T)
    return Q.T
