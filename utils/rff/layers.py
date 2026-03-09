import torch.nn as nn

from typing import Optional
from torch import Tensor
from . import functional

class BaseEncoding(nn.Module):
    """Base class for coordinate encoding layers"""
    
    def forward(self, v: Tensor) -> Tensor:
        """Apply encoding to input tensor v"""
        raise NotImplementedError


class GaussianEncoding(BaseEncoding):
    """Random Fourier feature mapping: gamma(v) = (cos(2pi B v), sin(2pi B v))"""

    def __init__(self, sigma: Optional[float] = None,
                 input_size: Optional[float] = None,
                 encoded_size: Optional[float] = None,
                 b: Optional[Tensor] = None):
        super().__init__()
        
        # Validate inputs
        if b is None:
            if None in (sigma, input_size, encoded_size):
                raise ValueError('sigma, input_size, encoded_size required when b is None')
            b = functional.sample_b(sigma, (encoded_size, input_size))
        elif any(arg is not None for arg in (sigma, input_size, encoded_size)):
            raise ValueError('Only specify b when using it')
            
        self.b = nn.parameter.Parameter(b, requires_grad=False)

    def forward(self, v: Tensor) -> Tensor:
        return functional.gaussian_encoding(v, self.b)


class BasicEncoding(BaseEncoding):
    """Basic encoding: gamma(v) = (cos(2pi v), sin(2pi v))"""

    def forward(self, v: Tensor) -> Tensor:
        return functional.basic_encoding(v)


class PositionalEncoding(BaseEncoding):
    """Positional encoding with multiple frequencies"""

    def __init__(self, sigma: float, m: int):
        super().__init__()
        self.sigma = sigma
        self.m = m

    def forward(self, v: Tensor) -> Tensor:
        return functional.positional_encoding(v, v, self.m)