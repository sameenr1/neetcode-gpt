class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        x = init

        for i in range (0, iterations):
            f_prime = 2*x
            x = x - learning_rate * f_prime

        return round(x, 5)
        # Round final answer to 5 decimal places
        
