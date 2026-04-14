import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.data_generator import generate_reservoir_data
from src.qubo_builder import QUBOBuilder
from src.classical.simulated_annealing import simulated_annealing
from src.classical.exact_solver import exact_solver
from src.classical.random_baseline import random_baseline


def evaluate_solution(x, Q, production, interference):
    energy = float(x @ Q @ x)
    total_production = float(np.sum(production * x))
    interference_penalty = float(np.sum(interference * np.outer(x, x)))
    return energy, total_production, interference_penalty


def run_single_experiment(n_wells=10, seed=42):
    porosity, perm, pressure, dist = generate_reservoir_data(n_wells=n_wells, seed=seed)
    production, interference = QUBOBuilder.from_reservoir_properties(
        porosity, perm, pressure, dist
    )

    Q = QUBOBuilder(production, interference).build_qubo()

    results = []

    methods = {
        "Random": lambda: random_baseline(Q, trials=1000),
        "SA": lambda: simulated_annealing(Q, max_iter=2000),
        "Exact": lambda: exact_solver(Q),
    }

    for method_name, solver in methods.items():
        start = time.perf_counter()
        result = solver()
        runtime = time.perf_counter() - start

        x = result["x"]
        energy, total_production, interference_penalty = evaluate_solution(
            x, Q, production, interference
        )

        results.append({
            "method": method_name,
            "n_wells": n_wells,
            "seed": seed,
            "energy": energy,
            "production": total_production,
            "interference": interference_penalty,
            "runtime_sec": runtime,
            "selected_wells": ",".join(map(str, np.where(x == 1)[0].tolist()))
        })

    return results


def run_benchmark():
    all_results = []

    for n_wells in [6, 8, 10]:
        for seed in range(10):
            all_results.extend(run_single_experiment(n_wells=n_wells, seed=seed))

    df = pd.DataFrame(all_results)
    df.to_csv("comparison_results.csv", index=False)

    summary = df.groupby(["method", "n_wells"], as_index=False).agg({
        "energy": ["mean", "std"],
        "production": ["mean", "std"],
        "interference": ["mean", "std"],
        "runtime_sec": ["mean", "std"],
    })

    summary.columns = [
        "method", "n_wells",
        "energy_mean", "energy_std",
        "production_mean", "production_std",
        "interference_mean", "interference_std",
        "runtime_mean", "runtime_std"
    ]
    summary.to_csv("comparison_summary.csv", index=False)

    print("\n=== Comparison Summary ===")
    print(summary)

    return df, summary


def plot_metric(summary, metric, ylabel, filename, lower_is_better=False):
    plt.figure(figsize=(8, 5))

    methods = summary["method"].unique()
    n_values = sorted(summary["n_wells"].unique())
    x = np.arange(len(n_values))
    width = 0.25

    offsets = np.linspace(-width, width, len(methods))

    for i, method in enumerate(methods):
        subset = summary[summary["method"] == method].sort_values("n_wells")
        plt.bar(
            x + offsets[i],
            subset[f"{metric}_mean"],
            width=width,
            yerr=subset[f"{metric}_std"],
            capsize=4,
            label=method
        )

    plt.xticks(x, n_values)
    plt.xlabel("Number of Candidate Wells")
    plt.ylabel(ylabel)
    title_note = "Lower is better" if lower_is_better else "Higher is better"
    plt.title(f"{ylabel} Comparison ({title_note})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


if __name__ == "__main__":
    df, summary = run_benchmark()

    plot_metric(summary, "energy", "Objective Energy", "energy_comparison.png", lower_is_better=True)
    plot_metric(summary, "production", "Production", "production_comparison.png", lower_is_better=False)
    plot_metric(summary, "interference", "Interference Penalty", "interference_comparison.png", lower_is_better=True)
    plot_metric(summary, "runtime_mean".replace("_mean", ""), "Runtime (sec)", "runtime_comparison.png", lower_is_better=True)
