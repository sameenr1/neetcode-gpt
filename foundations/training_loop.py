import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))

        w = np.zeros(np.size(X, axis = 1))
        b = 0
        n = np.size(X, axis = 0)

        for i in range (epochs):
            y_hat = X @ w + b

            dl_dw = (2/n)*np.transpose(X) @ (y_hat-y)
            dl_db = (2/n)*np.sum(y_hat - y)

            w = w - lr*dl_dw
            b = b - lr*dl_db

        return (np.round(w, 5), round(b, 5))
