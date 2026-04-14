import warnings
import numpy as np


class UltimateQUBOBuilder:
    def __init__(self, grid: dict, economic_params: dict):
        """
        grid: dictionary containing:
            - permeability
            - thickness
            - distance_matrix
            - porosity (optional)
            - pressure (optional)
            - drainage_radius (optional)
            - radius_investigation (optional)

        economic_params: dictionary containing:
            - well_cost
            - oil_price
            - discount_rate
            - project_life
        """
        self.grid = grid
        self.well_cost = economic_params.get("well_cost", 2e6)
        self.oil_price = economic_params.get("oil_price", 75)
        self.discount_rate = economic_params.get("discount_rate", 0.10)
        self.project_life = economic_params.get("project_life", 10)

    def productivity_index(self) -> np.ndarray:
        """
        Peaceman-inspired productivity index approximation.
        PI = 2πkh / (μ B ln(re/rw))
        Then multiplied by assumed pressure drawdown.
        """
        permeability = np.asarray(self.grid["permeability"], dtype=float)
        thickness = np.asarray(self.grid["thickness"], dtype=float)

        kh = permeability * thickness
        mu = 1.0
        B = 1.2
        r_e = float(self.grid.get("drainage_radius", 1000))
        r_w = 0.3

        PI = 2 * np.pi * kh / (mu * B * np.log(r_e / r_w))

        delta_p = 500.0
        return PI * delta_p

    def interference_matrix(self) -> np.ndarray:
        """
        Exponential distance-based interference penalty.
        """
        dist = np.asarray(self.grid["distance_matrix"], dtype=float)
        r_inv = float(self.grid.get("radius_investigation", 500))

        I = np.exp(-dist / r_inv)
        np.fill_diagonal(I, 0.0)
        return I

    def build_qubo(self, lambda_interf: float = 1.0):
        P = self.productivity_index()
        I = self.interference_matrix()
        n = len(P)

        annual_revenue = self.oil_price * 365 * P
        npv_factor = sum(
            (1 / (1 + self.discount_rate) ** t)
            for t in range(1, self.project_life + 1)
        )
        revenue_npv = annual_revenue * npv_factor

        Q = np.zeros((n, n), dtype=float)

        for i in range(n):
            Q[i, i] = -(revenue_npv[i] - self.well_cost)

        for i in range(n):
            for j in range(i + 1, n):
                Q[i, j] = lambda_interf * I[i, j]
                Q[j, i] = Q[i, j]

        metadata = {
            "P": P,
            "I": I,
            "revenue_npv": revenue_npv,
            "well_cost": self.well_cost,
            "oil_price": self.oil_price,
            "discount_rate": self.discount_rate,
            "project_life": self.project_life,
        }
        return Q, metadata


class QUBOBuilder:
    """
    Deprecated compatibility class.
    Kept temporarily so the old code does not break immediately.
    """

    def __init__(self, production: np.ndarray, interference: np.ndarray):
        warnings.warn(
            "QUBOBuilder is deprecated. Use UltimateQUBOBuilder instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.production = np.asarray(production, dtype=float)
        self.interference = np.asarray(interference, dtype=float)
        self.n = len(self.production)

    def build_qubo(self) -> np.ndarray:
        Q = np.zeros((self.n, self.n), dtype=float)

        for i in range(self.n):
            Q[i, i] = -self.production[i]

        for i in range(self.n):
            for j in range(i + 1, self.n):
                Q[i, j] = self.interference[i, j]
                Q[j, i] = self.interference[j, i]

        return Q

    @staticmethod
    def from_reservoir_properties(
        porosity: np.ndarray,
        permeability: np.ndarray,
        pressure: np.ndarray,
        distance_matrix: np.ndarray,
        alpha: float = 0.7,
        beta: float = 0.3,
    ):
        porosity = np.asarray(porosity, dtype=float)
        permeability = np.asarray(permeability, dtype=float)
        pressure = np.asarray(pressure, dtype=float)
        distance_matrix = np.asarray(distance_matrix, dtype=float)

        por_norm = (porosity - porosity.min()) / (
            porosity.max() - porosity.min() + 1e-8
        )
        perm_norm = (permeability - permeability.min()) / (
            permeability.max() - permeability.min() + 1e-8
        )
        pres_norm = (pressure - pressure.min()) / (
            pressure.max() - pressure.min() + 1e-8
        )

        production = (
            alpha * por_norm
            + beta * perm_norm
            + (1 - alpha - beta) * pres_norm
        )

        interference = 1.0 / (1.0 + distance_matrix)
        np.fill_diagonal(interference, 0.0)

        return production, interference