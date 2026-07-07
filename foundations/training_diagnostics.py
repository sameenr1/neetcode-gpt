import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []

        with torch.no_grad():

            current = x

            for layer in model:

                current = layer(current)

                if isinstance(layer, nn.Linear):

                    # 1. mean and std over ALL activations
                    mean_val = current.mean()
                    std_val = current.std()

                    # 2. dead neurons (check per column/neuron)
                    # shape: (batch_size, num_neurons)

                    dead_neurons = (current <= 0).all(dim=0)  # shape: (num_neurons,)

                    dead_fraction = dead_neurons.float().mean()

                    # 3. store results
                    stats.append({
                        "mean": round(mean_val.item(), 4),
                        "std": round(std_val.item(), 4),
                        "dead_fraction": round(dead_fraction.item(), 4)
                    })

        return stats
            

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.

        stats = []
        model.zero_grad()
        y_hat = model(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        loss.backward()

        for layer in model.modules():
            if isinstance (layer, nn.Linear):
                g = layer.weight.grad

                if g is not None:
                    mean = g.mean()
                    std = g.std()
                    norm = g.norm()


                    stats.append({
                        "mean": round(mean.item(), 4),
                        "std": round(std.item(), 4),
                        "norm": round(norm.item(), 4)
                     })

        return stats


    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)

        for layer in activation_stats:
            if layer["dead_fraction"] > 0.5:
                return "dead_neurons"
        
        for layer in gradient_stats:
            if layer["norm"] > 1000:
                return "exploding_gradients"

        if gradient_stats[-1]["norm"] < 10**(-5):
            return "vanishing_gradients"

        for layer in activation_stats:
            if layer["std"] < 0.1:
                 return "vanishing_gradients"
            elif layer["std"] > 10.0:
                return "exploding_gradients"
            else:
                continue

        return "healthy"
        

        
