import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                             x: List[float],
                             W1: List[List[float]], b1: List[float],
                             W2: List[List[float]], b2: List[float],
                             y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)

        # 1. Convert all inputs to NumPy arrays and force them to be 2D arrays 
        # to ensure matrix operations behave predictably regardless of batch size.
        x = np.array(x).reshape(1, -1)
        W1 = np.array(W1)
        b1 = np.array(b1).reshape(1, -1)
        W2 = np.array(W2)
        b2 = np.array(b2).reshape(1, -1)
        y_true = np.array(y_true).reshape(1, -1)

        # 2. Forward Pass
        z1 = x @ W1.T + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2.T + b2
        
        # Get the batch size n
        n = y_true.shape[0]

        # 3. Compute Loss
        loss = np.mean((z2 - y_true)**2)

        # 4. Backward Pass
        dl_dz2 = (2 / n) * (z2 - y_true)        # Shape: (batch_size, output_dim)
        dl_dw2 = dl_dz2.T @ a1                  # Shape: (output_dim, hidden_dim)
        dl_db2 = np.sum(dl_dz2, axis=0)         # Sum across batch to make it 1D

        # Fast, vectorized ReLU binary mask
        binary_mask = (z1 > 0).astype(float)    # Shape: (batch_size, hidden_dim)

        dl_dz1 = (dl_dz2 @ W2) * binary_mask    # Shape: (batch_size, hidden_dim)
        dl_dw1 = dl_dz1.T @ x                   # Shape: (hidden_dim, input_dim)
        dl_db1 = np.sum(dl_dz1, axis=0)         # Sum across batch to make it 1D

        # 5. Round results to 4 decimals and convert back to pure Python types
        output = {
            'loss': float(np.round(loss, 4)),
            'dW1': np.round(dl_dw1, 4).tolist(),
            'db1': np.round(dl_db1, 4).tolist(),
            'dW2': np.round(dl_dw2, 4).tolist(),
            'db2': np.round(dl_db2, 4).tolist()
        }
        
        return output