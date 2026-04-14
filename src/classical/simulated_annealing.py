import numpy as np

def simulated_annealing(Q, max_iter=1000, temp_init=10.0, temp_min=0.01, alpha=0.995):
    n = Q.shape[0]
    x = np.random.randint(0, 2, n)
    best_x = x.copy()
    best_energy = x @ Q @ x
    temp = temp_init

    for _ in range(max_iter):
        new_x = x.copy()
        flip = np.random.randint(n)
        new_x[flip] = 1 - new_x[flip]
        new_energy = new_x @ Q @ new_x
        delta = new_energy - best_energy

        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            x = new_x
            if new_energy < best_energy:
                best_energy = new_energy
                best_x = new_x.copy()

        temp = max(temp_min, temp * alpha)

    return {"x": best_x, "energy": float(best_energy)}
