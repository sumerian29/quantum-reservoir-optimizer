import numpy as np

class QUBOBuilder:
    def __init__(self, production, interference):
        self.n = len(production)
        self.P = production
        self.I = interference

    def build_qubo(self):
        Q = np.zeros((self.n, self.n))
        for i in range(self.n):
            Q[i, i] = -self.P[i]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                Q[i, j] = self.I[i, j]
                Q[j, i] = self.I[j, i]
        return Q

    @staticmethod
    def from_reservoir_properties(porosity, permeability, pressure, distance_matrix, alpha=0.4, beta=0.3):
        por_norm = (porosity - porosity.min()) / (porosity.max() - porosity.min() + 1e-8)
        perm_norm = (permeability - permeability.min()) / (permeability.max() - permeability.min() + 1e-8)
        pres_norm = (pressure - pressure.min()) / (pressure.max() - pressure.min() + 1e-8)

        production = alpha * por_norm + beta * perm_norm + (1 - alpha - beta) * pres_norm
        interference = 1.0 / (1.0 + distance_matrix)
        np.fill_diagonal(interference, 0.0)

        return production, interference
