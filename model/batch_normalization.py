import numpy as np
import torch
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists


        x_t = torch.tensor(x)
        gamma_t = torch.tensor(gamma)
        beta_t = torch.tensor(beta)
        running_mean_t = torch.tensor(running_mean)
        running_var_t = torch.tensor(running_var)

        N = x_t.size(0)
        mean = torch.mean (x_t, dim = 0)
        var = (1/N)*torch.sum((x_t-mean)**2, dim = 0)

        
        if training == True:
            x_hat = (x_t-mean)/(torch.sqrt(var+eps))
            out = torch.round(gamma_t*x_hat + beta_t, decimals = 4)
            running_mean_t = torch.round((1-momentum)*running_mean_t+momentum*mean, decimals = 4)
            running_var_t = torch.round((1-momentum)*running_var_t + momentum*var, decimals = 4)
        else:
            x_hat = (x_t-running_mean_t)/(torch.sqrt(running_var_t+eps))
            out = torch.round(gamma_t*x_hat + beta_t, decimals = 4)
            
        return (
    [[round(v, 4) for v in row] for row in out.tolist()],
    [round(v, 4) for v in running_mean_t.tolist()],
    [round(v, 4) for v in running_var_t.tolist()]
)



