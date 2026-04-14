import yaml
import numpy as np

from src.utils.plots import plot_well_selection, plot_production_vs_selection
from src.utils.data_generator import generate_reservoir_data
from src.qubo_builder import QUBOBuilder
from src.classical.simulated_annealing import simulated_annealing


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    porosity, perm, pressure, dist = generate_reservoir_data(
        n_wells=config["n_wells"]
    )

    production, interference = QUBOBuilder.from_reservoir_properties(
        porosity, perm, pressure, dist
    )

    builder = QUBOBuilder(production, interference)
    Q = builder.build_qubo()

    result = simulated_annealing(Q)

    print("Selected wells:", np.where(result["x"] == 1)[0].tolist())
    print("Objective value:", result["energy"])
    print("Estimated production:", float(np.sum(production * result["x"])))

    plot_well_selection(result["x"], production)
    plot_production_vs_selection(result["x"], production)


if __name__ == "__main__":
    main()
