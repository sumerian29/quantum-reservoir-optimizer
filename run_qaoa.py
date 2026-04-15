import csv
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from src.core.qubo_builder import UltimateQUBOBuilder


def qubo_to_ising(Q: np.ndarray) -> tuple[SparsePauliOp, float]:
    """
    Convert a QUBO matrix into an Ising Hamiltonian plus a constant offset.
    """
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

    hamiltonian = SparsePauliOp(pauli_strings, coeffs)
    return hamiltonian, const


def apply_cost_layer(qc: QuantumCircuit, gamma: float, hamiltonian: SparsePauliOp):
    """
    Apply exp(-i * gamma * H_cost) using RZ and CX gates
    for 1-local and 2-local Z terms.
    """
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


def build_qaoa_circuit(
    hamiltonian: SparsePauliOp,
    params: np.ndarray,
    p: int
) -> QuantumCircuit:
    """
    Build the final QAOA circuit from optimized angles.

    params format:
        [gamma_1, ..., gamma_p, beta_1, ..., beta_p]
    """
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


def bitstring_to_array(bitstring: str) -> np.ndarray:
    """
    Convert a measured bitstring into a 0/1 array.
    Qiskit bitstrings are usually displayed in reverse order
    relative to qubit indexing, so reverse here.
    """
    return np.array([int(b) for b in bitstring[::-1]], dtype=int)


def evaluate_qubo_bitstring(Q: np.ndarray, bitstring: str) -> float:
    """
    Evaluate the QUBO objective for a given bitstring.
    """
    x = bitstring_to_array(bitstring)
    return float(x @ Q @ x)


def sample_best_bitstring(qc: QuantumCircuit, shots: int = 4096) -> tuple[str, dict]:
    """
    Sample the final QAOA circuit and return the most frequent bitstring.
    """
    simulator = AerSimulator()
    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts()
    best_bitstring = max(counts, key=counts.get)
    return best_bitstring, counts


def qaoa_manual(
    hamiltonian: SparsePauliOp,
    const: float,
    p: int = 2,
    maxiter: int = 100
) -> tuple[float, np.ndarray]:
    """
    Run a manual QAOA optimization using Aer Estimator and COBYLA.

    Returns:
        optimal_energy, optimal_params
    """
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
        np.random.uniform(0, np.pi, p),
        np.random.uniform(0, np.pi, p)
    ])

    res = minimize(
        objective,
        initial_params,
        method="COBYLA",
        options={"maxiter": maxiter}
    )

    return res.fun, res.x


def save_outputs(
    output_dir: Path,
    optimal_energy: float,
    optimal_params: np.ndarray,
    best_bitstring: str,
    best_x: np.ndarray,
    selected_wells: list[int],
    best_qubo_value: float,
    counts: dict,
    metadata: dict,
):
    """
    Save clean report-ready outputs.
    """
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "qaoa_final_result.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "optimal_energy",
            "best_bitstring",
            "binary_decision",
            "selected_wells",
            "qubo_value",
        ])
        writer.writerow([
            optimal_energy,
            best_bitstring,
            " ".join(map(str, best_x.tolist())),
            " ".join(map(str, selected_wells)),
            best_qubo_value,
        ])

    txt_path = output_dir / "qaoa_engineering_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Quantum Reservoir Optimizer - Final Engineering Summary\n")
        f.write("=====================================================\n\n")
        f.write(f"Optimal Energy from optimizer: {optimal_energy:.6f}\n")
        f.write(f"Optimal Angles (gamma, beta): {optimal_params.tolist()}\n")
        f.write(f"Best Bitstring: {best_bitstring}\n")
        f.write(f"Binary Decision (0/1): {best_x.tolist()}\n")
        f.write(f"Selected Wells: {selected_wells}\n")
        f.write(f"QUBO Value of Selected Solution: {best_qubo_value:.6f}\n\n")
        f.write(f"Metadata Keys: {list(metadata.keys())}\n\n")
        f.write("Top Measurement Counts:\n")
        for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]:
            f.write(f"{k} : {v}\n")

    return csv_path, txt_path


def main():
    # Synthetic reservoir input data
    grid = {
        "permeability": np.array([100, 150, 200], dtype=float),
        "thickness": np.array([50, 55, 60], dtype=float),
        "distance_matrix": np.array(
            [
                [0, 300, 500],
                [300, 0, 250],
                [500, 250, 0],
            ],
            dtype=float,
        ),
        "drainage_radius": 1000,
        "radius_investigation": 500,
    }

    # Economic parameters
    econ = {
        "well_cost": 2e6,
        "oil_price": 75,
        "discount_rate": 0.10,
        "project_life": 10,
    }

    p = 2

    # Build QUBO from physics + economics
    builder = UltimateQUBOBuilder(grid, econ)
    Q, metadata = builder.build_qubo()

    print("\n========== FINAL RESULTS ==========\n")

    print("QUBO Matrix:")
    print(Q)
    print()

    # Convert to Ising form
    hamiltonian, const = qubo_to_ising(Q)

    print(f"Ising Hamiltonian (offset = {const:.4f}):")
    print(hamiltonian)
    print()

    # Run QAOA optimization
    print(f"Running QAOA optimization (p={p}, COBYLA)...")
    optimal_energy, optimal_params = qaoa_manual(
        hamiltonian,
        const,
        p=p,
        maxiter=80,
    )

    print(f"Optimal Energy from optimizer: {optimal_energy:.6f}")
    print(f"Optimal Angles (gamma, beta): {optimal_params}")
    print()

    # Build final measured circuit and sample bitstrings
    final_qc = build_qaoa_circuit(hamiltonian, optimal_params, p=p)
    best_bitstring, counts = sample_best_bitstring(final_qc, shots=4096)

    # Convert best bitstring into a binary decision vector
    best_x = bitstring_to_array(best_bitstring)
    best_qubo_value = evaluate_qubo_bitstring(Q, best_bitstring)

    print("Best Bitstring:", best_bitstring)
    print("Binary Decision (0/1):", best_x.tolist())

    selected_wells = [i for i, v in enumerate(best_x) if v == 1]
    print("Selected Wells:", selected_wells)
    print(f"QUBO Value of Selected Solution: {best_qubo_value:.6f}")
    print()

    print("Metadata Keys:")
    print(list(metadata.keys()))
    print()

    print("Top Measurement Counts:")
    for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]:
        print(f"{k} : {v}")

    print()

    # Save outputs
    output_dir = Path("results")
    csv_path, txt_path = save_outputs(
        output_dir=output_dir,
        optimal_energy=optimal_energy,
        optimal_params=optimal_params,
        best_bitstring=best_bitstring,
        best_x=best_x,
        selected_wells=selected_wells,
        best_qubo_value=best_qubo_value,
        counts=counts,
        metadata=metadata,
    )

    print(f"Saved CSV summary to: {csv_path}")
    print(f"Saved TXT summary to: {txt_path}")

    print("\n===================================\n")


if __name__ == "__main__":
    main()
