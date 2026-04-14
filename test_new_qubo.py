import numpy as np
from src.core.qubo_builder import UltimateQUBOBuilder

grid = {
    "permeability": np.array([100, 150, 200], dtype=float),
    "thickness": np.array([50, 55, 60], dtype=float),
    "distance_matrix": np.array([
        [0, 300, 500],
        [300, 0, 250],
        [500, 250, 0]
    ], dtype=float),
    "drainage_radius": 1000,
    "radius_investigation": 500,
}

econ = {
    "well_cost": 2e6,
    "oil_price": 75,
    "discount_rate": 0.10,
    "project_life": 10,
}

builder = UltimateQUBOBuilder(grid, econ)
Q, metadata = builder.build_qubo()

print("Q shape:", Q.shape)
print("Q matrix:")
print(Q)
print("Metadata keys:", list(metadata.keys()))