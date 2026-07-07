import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list

        x_n  = np.array (x)
        gamma_n = np.array(gamma)

        N = np.size(x_n)
        RMS = np.sqrt((1/N)*np.sum(x_n**2)+eps)
        x_hat = x_n/RMS

        out = gamma_n*x_hat
        output = out.tolist()

        output = [round(v, 4) for v in output]
        
        return output
        
