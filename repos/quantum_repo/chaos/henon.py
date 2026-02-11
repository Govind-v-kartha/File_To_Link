from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np 

def henon_map(x0, y0, alpha=1.8, beta=0.3, n_iter=64):
    x = np.zeros(n_iter)
    y = np.zeros(n_iter)
    x[0], y[0] = x0, y0
    for i in range(1, n_iter):
        x[i] = 1 - alpha * (x[i-1] ** 2) + y[i-1]
        y[i] = beta * x[i-1]
    return x, y