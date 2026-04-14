import numpy as np

def generate_reservoir_data(n_wells=10, seed=42):
    np.random.seed(seed)

    x = np.random.uniform(0, 100, n_wells)
    y = np.random.uniform(0, 100, n_wells)
    distance = np.sqrt((x[:, None] - x[None, :])**2 + (y[:, None] - y[None, :])**2)

    porosity = np.random.beta(2, 5, n_wells) * 0.3
    permeability = np.random.lognormal(mean=1, sigma=0.5, size=n_wells) * 100
    pressure = np.random.uniform(2000, 4000, n_wells)

    return porosity, permeability, pressure, distance
