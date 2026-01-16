import numpy as np
from itertools import product

def rosenbrock(x: float, y: float) -> float:
    return 100 * (y - x**2)**2 + (1 - x)**2

def rosenbrock_derivative_x(x, y) -> float:
    return -400 * x * (y - x**2) - 2 * (1 - x)

def rosenbrock_derivative_y(x, y) -> float:
    return 200 * (y - x**2)

def rosenbrock_gradient(x, y) -> np.ndarray:
    return np.array([
        rosenbrock_derivative_x(x, y),
        rosenbrock_derivative_y(x, y)
    ])

def rosenbrock_directional_derivative(x, y, p) -> float:
    grad = rosenbrock_gradient(x, y)
    return grad[0] * p[0] + grad[1] * p[1]

def crosstest(func: callable, *varrs) -> None:
    """Test function with all combinations (product) of variables as arguments"""
    for args in product(*varrs):
        func(*args)