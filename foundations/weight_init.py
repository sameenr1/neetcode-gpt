import math
from typing import List
import torch
import torch.nn as nn


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2 / (fan_in + fan_out))
        tensor_w = torch.round(torch.randn(fan_out, fan_in) * std, decimals=4)
        return tensor_w.tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2 / fan_in)
        tensor_w = torch.round(torch.randn(fan_out, fan_in) * std, decimals=4)
        return tensor_w.tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # 1. Set seed at the absolute beginning
        torch.manual_seed(0)

        # Track the input/output dimension layout across the sequential chain
        dims = [input_dim] + [hidden_dim] * num_layers
        weights = []

        # 2. Generate ALL weight matrices sequentially first to preserve RNG state order
        for i in range(num_layers):
            fan_in = dims[i]
            fan_out = dims[i + 1]

            if init_type == 'xavier':
                std = math.sqrt(2 / (fan_in + fan_out))
            elif init_type == 'kaiming':
                std = math.sqrt(2 / fan_in)
            elif init_type == 'random':
                std = 1.0
            else:
                raise ValueError(f"Unknown initialization type: {init_type}")

            w = torch.randn(fan_out, fan_in) * std
            weights.append(w)

        # 3. Generate your random input vector x AFTER all weights are constructed
        x = torch.randn(1, input_dim)
        std_history = []

        # 4. Now run your forward passes sequentially through the pre-built chain
        for w in weights:
            x = x @ w.T
            x = torch.relu(x)
            std_history.append(round(x.std().item(), 2))

        return std_history