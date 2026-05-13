"""Compute J = ∂f(x; θ) / ∂θ for both architectures."""
from __future__ import annotations

import numpy as np
import torch
from torch.func import functional_call, jacrev, vmap

from models import (MLP, CNN, mlp_vector_to_params, cnn_vector_to_state)


def _jacobian_factory(model, x_batch, flat, vec2state_fn):
    """Return per-sample Jacobian via torch.func vmap+jacrev."""

    def single_output(p: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        sd = vec2state_fn(p)
        out = functional_call(model, sd, (xi.unsqueeze(0) if xi.dim() == 3 else xi,))
        return out.reshape(-1)

    def per_sample(p, xi):
        return jacrev(single_output, argnums=0)(p, xi)

    return vmap(per_sample, in_dims=(None, 0))(flat, x_batch)


def compute_jacobian_mlp(model: MLP, x: torch.Tensor,
                          flat: torch.Tensor) -> torch.Tensor:
    """J: (n*d_out, P) on the same device as `flat`."""
    Js = _jacobian_factory(model, x, flat, mlp_vector_to_params)
    return Js.reshape(-1, flat.numel()).detach()


def compute_jacobian_cnn(model: CNN, X: torch.Tensor, flat: torch.Tensor,
                          chunk_size: int = 100) -> torch.Tensor:
    """Chunked Jacobian on CPU; (n*d_out, P) float32."""
    c1, c2, fc_out = model.c1, model.c2, model.fc_out

    def vec2state(p):
        return cnn_vector_to_state(p, c1, c2, fc_out)

    out = []
    n = X.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        Jc = _jacobian_factory(model, X[start:end], flat, vec2state)
        out.append(Jc.reshape(-1, flat.numel()).detach().cpu())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(out, dim=0)
