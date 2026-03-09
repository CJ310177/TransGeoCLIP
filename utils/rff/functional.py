import numpy as np
import torch
from torch import Tensor

# Constants
TWO_PI = 2 * np.pi


def sample_b(sigma: float, size: tuple) -> Tensor:
    """Sample matrix from N(0, sigma^2)"""
    return torch.randn(size) * sigma


def _apply_encoding(v: Tensor, vp: Tensor) -> Tensor:
    """Apply cos/sin encoding to projected input"""
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


@torch.jit.script
def gaussian_encoding(v: Tensor, b: Tensor) -> Tensor:
    """gamma(v) = (cos(2pi B v), sin(2pi B v))"""
    vp = TWO_PI * v @ b.T
    return _apply_encoding(v, vp)


@torch.jit.script
def basic_encoding(v: Tensor) -> Tensor:
    """gamma(v) = (cos(2pi v), sin(2pi v))"""
    vp = TWO_PI * v
    return _apply_encoding(v, vp)


@torch.jit.script
def positional_encoding(v: Tensor, sigma: float, m: int) -> Tensor:
    """gamma(v) = (cos(2pi sigma^(j/m) v), sin(2pi sigma^(j/m) v)), j=0..m-1"""
    j = torch.arange(m, device=v.device)
    coeffs = TWO_PI * sigma ** (j / m)
    vp = coeffs * torch.unsqueeze(v, -1)
    return _apply_encoding(v, vp).flatten(-2, -1)