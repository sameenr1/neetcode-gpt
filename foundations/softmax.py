import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        calc_1 = z - max(z)
        softmax = np.exp(calc_1)/(np.sum(np.exp(calc_1)))
        return np.round(softmax, 4)
        
