import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        z = np.dot(x, w) + b 
        y_hat =1/(1 + np.exp(-z))
        L = 0.5 * (y_hat - y_true)**2

        da_dz = np.exp(-z)/((1+np.exp(-z))**2)
        dL_da = y_hat - y_true
        dz_dw = x
        dz_db = 1
        dL_dw = dL_da*da_dz*dz_dw
        dL_db = dL_da*da_dz

        return (np.round(dL_dw, 5), np.round(dL_db, 5))

        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
       
