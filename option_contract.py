import numpy as np
from dataclasses import dataclass

@dataclass
class OptionContract:
    S: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: str # 'call' or 'put'
    
    def __post_init__(self):
        self.option_type = self.option_type.lower()
        if self.option_type not in ('call', 'put'):
            raise ValueError("option_type must be 'call' or 'put'")
        if self.S <= 0 or self.K <= 0 or self.T <= 0 or self.sigma <= 0:
            raise ValueError("S, K, T, and sigma must be positive")
        if not all(np.isfinite(x) for x in [self.S, self.K, self.T, self.r, self.sigma]):
            raise ValueError("All inputs must be finite numbers")
    