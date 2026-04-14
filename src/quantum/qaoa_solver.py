import numpy as np
from itertools import product

def exact_solver(Q):
    n = Q.shape[0]
    best_x = None
    best_energy = float("inf")

    for bits in product([0, 1], repeat=n):
        x = np.array(bits, dtype=int)
        energy = float(x @ Q @ x)
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()

    return {"x": best_x, "energy": best_energy}
