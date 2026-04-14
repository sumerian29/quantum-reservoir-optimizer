import numpy as np

def random_baseline(Q, trials=500):
    n = Q.shape[0]
    best_x = None
    best_energy = float("inf")

    for _ in range(trials):
        x = np.random.randint(0, 2, n)
        energy = float(x @ Q @ x)
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()

    return {"x": best_x, "energy": best_energy}
