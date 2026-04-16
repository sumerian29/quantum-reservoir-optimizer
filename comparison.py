import csv
import time
from itertools import product
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator
from qiskit_aer import AerSimulator


# ============================================================
# 1) Synthetic reservoir case generator
# ============================================================

def generate_synthetic_case(n_wells: int, seed: int = 0):
    rng = np.random.default_rng(seed)

    # Production potential for each candidate well
    production = rng.uniform(0.8, 1.6, size=n_wells)

    # Random 2D positions for wells
    coords = rng.uniform(0, 1000, size=(n_wells, 2))

    # Distance matrix
    dist = np.zeros((n_wells, n_wells))
    for i in range(n_wells):
        for j in range(n_wells):
            dist[i, j] = np.linalg.norm(coords[i] - coords[j])

    # Interference penalty: stronger when wells are close
    scale = 400.0
    interference = np.exp(-dist / scale)
    np.fill_diagonal(interference, 0.0)

    return production, interference


def build_qubo(production: np.ndarray, interference: np.ndarray, lam: float = 1.0):
    n = len(production)
    Q = np.zeros((n, n), dtype=float)

    # Reward selected wells -> negative diagonal for minimization
    for i in range(n):
        Q[i, i] = -production[i]

    # Penalize selecting interfering pairs
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = lam * interference[i, j]
            Q[j, i] = Q[i, j]

    return Q


def evaluate_solution(x: np.ndarray, production: np.ndarray, interference: np.ndarray, Q: np.ndarray):
    energy = float(x @ Q @ x)
    prod_value = float(np.sum(production * x))

    penalty = 0.0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            penalty += interference[i, j] * x[i] * x[j]

    return energy, prod_value, float(penalty)


# ============================================================
# 2) Classical solvers
# ============================================================

def exact_solver(Q: np.ndarray):
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


def random_solver(Q: np.ndarray, n_samples: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = Q.shape[0]

    best_x = None
    best_energy = float("inf")

    for _ in range(n_samples):
        x = rng.integers(0, 2, size=n)
        energy = float(x @ Q @ x)
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()

    return {"x": best_x, "energy": best_energy}


def simulated_annealing(Q: np.ndarray, n_steps: int = 4000, temp_start: float = 5.0, temp_end: float = 0.01, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = Q.shape[0]

    x = rng.integers(0, 2, size=n)
    energy = float(x @ Q @ x)

    best_x = x.copy()
    best_energy = energy

    for step in range(n_steps):
        temp = temp_start * ((temp_end / temp_start) ** (step / max(1, n_steps - 1)))

        candidate = x.copy()
        idx = rng.integers(0, n)
        candidate[idx] = 1 - candidate[idx]

        candidate_energy = float(candidate @ Q @ candidate)
        delta = candidate_energy - energy

        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-12)):
            x = candidate
            energy = candidate_energy

            if energy < best_energy:
                best_energy = energy
                best_x = x.copy()

    return {"x": best_x, "energy": best_energy}


# ============================================================
# 3) QAOA helpers
# ============================================================

def qubo_to_ising(Q: np.ndarray):
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    const = 0.0

    for i in range(n):
        h[i] = Q[i, i] / 2.0
        const += Q[i, i] / 4.0
        for j in range(i + 1, n):
            J[i, j] = Q[i, j] / 4.0
            J[j, i] = J[i, j]
            const += Q[i, j] / 4.0

    pauli_strings = []
    coeffs = []

    for i in range(n):
        if h[i] != 0:
            z = ["I"] * n
            z[i] = "Z"
            pauli_strings.append("".join(z))
            coeffs.append(h[i])

    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0:
                z = ["I"] * n
                z[i] = "Z"
                z[j] = "Z"
                pauli_strings.append("".join(z))
                coeffs.append(J[i, j])

    return SparsePauliOp(pauli_strings, coeffs), const


def apply_cost_layer(qc: QuantumCircuit, gamma: float, hamiltonian: SparsePauliOp):
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        label = pauli.to_label()

        if label == "I" * len(label):
            continue

        angle = 2.0 * gamma * float(np.real(coeff))
        qubits = [i for i, p in enumerate(label) if p == "Z"]

        if len(qubits) == 1:
            qc.rz(angle, qubits[0])

        elif len(qubits) == 2:
            q0, q1 = qubits
            qc.cx(q0, q1)
            qc.rz(angle, q1)
            qc.cx(q0, q1)

        else:
            raise ValueError(f"Unsupported Pauli term with >2 qubits: {label}")


def build_qaoa_circuit(hamiltonian: SparsePauliOp, params: np.ndarray, p: int):
    n_qubits = hamiltonian.num_qubits
    gamma_vals = params[:p]
    beta_vals = params[p:]

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    for layer in range(p):
        apply_cost_layer(qc, gamma_vals[layer], hamiltonian)
        for i in range(n_qubits):
            qc.rx(2.0 * beta_vals[layer], i)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def bitstring_to_array(bitstring: str):
    return np.array([int(b) for b in bitstring[::-1]], dtype=int)


def sample_best_bitstring(qc: QuantumCircuit, shots: int = 2048):
    simulator = AerSimulator()
    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts()
    best_bitstring = max(counts, key=counts.get)
    return best_bitstring, counts


def qaoa_manual(hamiltonian: SparsePauliOp, const: float, p: int = 2, maxiter: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_qubits = hamiltonian.num_qubits
    estimator = Estimator()

    def objective(params):
        gamma_vals = params[:p]
        beta_vals = params[p:]

        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))

        for layer in range(p):
            apply_cost_layer(qc, gamma_vals[layer], hamiltonian)
            for i in range(n_qubits):
                qc.rx(2.0 * beta_vals[layer], i)

        job = estimator.run([qc], [hamiltonian])
        result = job.result()
        energy = float(np.real(result.values[0])) + const
        return energy

    initial_params = np.concatenate([
        rng.uniform(0, np.pi, p),
        rng.uniform(0, np.pi, p)
    ])

    res = minimize(
        objective,
        initial_params,
        method="COBYLA",
        options={"maxiter": maxiter}
    )

    return res.fun, res.x


def qaoa_solver(Q: np.ndarray, p: int = 2, maxiter: int = 60, shots: int = 2048, seed: int = 0):
    hamiltonian, const = qubo_to_ising(Q)
    optimizer_energy, optimizer_params = qaoa_manual(
        hamiltonian,
        const,
        p=p,
        maxiter=maxiter,
        seed=seed,
    )

    final_qc = build_qaoa_circuit(hamiltonian, optimizer_params, p=p)
    best_bitstring, counts = sample_best_bitstring(final_qc, shots=shots)
    best_x = bitstring_to_array(best_bitstring)
    sampled_energy = float(best_x @ Q @ best_x)

    return {
        "x": best_x,
        "energy": sampled_energy,
        "optimizer_energy": optimizer_energy,
        "params": optimizer_params,
        "bitstring": best_bitstring,
        "counts": counts,
    }


# ============================================================
# 4) Benchmark runner
# ============================================================

def run_single_experiment(n_wells: int, seed: int):
    production, interference = generate_synthetic_case(n_wells=n_wells, seed=seed)
    Q = build_qubo(production, interference, lam=1.0)

    methods = {
        "Exact": lambda: exact_solver(Q),
        "QAOA": lambda: qaoa_solver(Q, p=2, maxiter=60, shots=2048, seed=seed),
        "Random": lambda: random_solver(Q, n_samples=2000, seed=seed),
        "SA": lambda: simulated_annealing(Q, n_steps=4000, seed=seed),
    }

    rows = []

    for method_name, solver in methods.items():
        t0 = time.perf_counter()
        result = solver()
        runtime = time.perf_counter() - t0

        x = result["x"]
        energy, prod_value, penalty = evaluate_solution(x, production, interference, Q)

        rows.append({
            "method": method_name,
            "n_wells": n_wells,
            "seed": seed,
            "energy": energy,
            "production": prod_value,
            "interference": penalty,
            "runtime_sec": runtime,
            "solution": "".join(map(str, x.tolist())),
        })

    return rows


def aggregate_results(rows):
    grouped = {}

    for row in rows:
        key = (row["method"], row["n_wells"])
        grouped.setdefault(key, {
            "energy": [],
            "production": [],
            "interference": [],
            "runtime_sec": [],
        })

        grouped[key]["energy"].append(row["energy"])
        grouped[key]["production"].append(row["production"])
        grouped[key]["interference"].append(row["interference"])
        grouped[key]["runtime_sec"].append(row["runtime_sec"])

    summary_rows = []
    for (method, n_wells), vals in grouped.items():
        summary_rows.append({
            "method": method,
            "n_wells": n_wells,
            "energy_mean": float(np.mean(vals["energy"])),
            "energy_std": float(np.std(vals["energy"])),
            "production_mean": float(np.mean(vals["production"])),
            "production_std": float(np.std(vals["production"])),
            "interference_mean": float(np.mean(vals["interference"])),
            "interference_std": float(np.std(vals["interference"])),
            "runtime_mean": float(np.mean(vals["runtime_sec"])),
            "runtime_std": float(np.std(vals["runtime_sec"])),
        })

    summary_rows.sort(key=lambda x: (x["n_wells"], x["method"]))
    return summary_rows


# ============================================================
# 5) Save CSV files
# ============================================================

def save_csv_results(rows, summary_rows, output_dir: Path):
    output_dir.mkdir(exist_ok=True)

    results_path = output_dir / "comparison_results.csv"
    summary_path = output_dir / "comparison_summary.csv"

    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "n_wells", "seed",
                "energy", "production", "interference", "runtime_sec", "solution"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "n_wells",
                "energy_mean", "energy_std",
                "production_mean", "production_std",
                "interference_mean", "interference_std",
                "runtime_mean", "runtime_std"
            ]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    return results_path, summary_path


# ============================================================
# 6) Plotting
# ============================================================

def plot_metric(summary_rows, metric_mean, metric_std, ylabel, title, filename, output_dir: Path):
    methods = ["Exact", "QAOA", "Random", "SA"]
    n_values = sorted(list({row["n_wells"] for row in summary_rows}))
    x = np.arange(len(n_values))
    width = 0.18

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, method in enumerate(methods):
        means = []
        stds = []

        for n in n_values:
            row = next(r for r in summary_rows if r["method"] == method and r["n_wells"] == n)
            means.append(row[metric_mean])
            stds.append(row[metric_std])

        ax.bar(
            x + (idx - 1.5) * width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=method
        )

    ax.set_xticks(x)
    ax.set_xticklabels(n_values)
    ax.set_xlabel("Number of Candidate Wells")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    plot_path = output_dir / filename
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    return plot_path


# ============================================================
# 7) Main
# ============================================================

def main():
    output_dir = Path("results")
    n_values = [6, 8, 10]
    seeds = [0, 1, 2]

    all_rows = []

    print("\n=== Running benchmark experiments ===\n")

    for n_wells in n_values:
        for seed in seeds:
            print(f"Running N={n_wells}, seed={seed} ...")
            rows = run_single_experiment(n_wells=n_wells, seed=seed)
            all_rows.extend(rows)

    summary_rows = aggregate_results(all_rows)

    results_path, summary_path = save_csv_results(all_rows, summary_rows, output_dir)

    energy_plot = plot_metric(
        summary_rows,
        metric_mean="energy_mean",
        metric_std="energy_std",
        ylabel="Objective Energy",
        title="Objective Energy Comparison (Lower is better)",
        filename="energy_comparison.png",
        output_dir=output_dir,
    )

    production_plot = plot_metric(
        summary_rows,
        metric_mean="production_mean",
        metric_std="production_std",
        ylabel="Production",
        title="Production Comparison (Higher is better)",
        filename="production_comparison.png",
        output_dir=output_dir,
    )

    interference_plot = plot_metric(
        summary_rows,
        metric_mean="interference_mean",
        metric_std="interference_std",
        ylabel="Interference Penalty",
        title="Interference Penalty Comparison (Lower is better)",
        filename="interference_comparison.png",
        output_dir=output_dir,
    )

    runtime_plot = plot_metric(
        summary_rows,
        metric_mean="runtime_mean",
        metric_std="runtime_std",
        ylabel="Runtime (sec)",
        title="Runtime Comparison (Lower is better)",
        filename="runtime_comparison.png",
        output_dir=output_dir,
    )

    print("\n=== Summary ===\n")
    for row in summary_rows:
        print(row)

    print("\nSaved files:")
    print(results_path)
    print(summary_path)
    print(energy_plot)
    print(production_plot)
    print(interference_plot)
    print(runtime_plot)
    print("\nBenchmark completed.\n")


if __name__ == "__main__":
    main()