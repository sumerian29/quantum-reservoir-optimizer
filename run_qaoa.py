from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer.primitives import Estimator


INPUT_FILE = Path("results") / "volve_qubo_inputs.npz"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_volve_qubo(npz_path: Path):
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing input file: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    Q = data["Q"].astype(float)
    well_names = data["well_names"].tolist()
    production = data["production"].astype(float)
    interference = data["interference_matrix"].astype(float)

    metadata = {
        "well_names": well_names,
        "production": production,
        "interference_matrix": interference,
        "source": "Volve-inspired field dataset",
    }
    return Q, metadata


def qubo_value(Q: np.ndarray, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(x @ Q @ x)


def brute_force_best_bitstring(Q: np.ndarray):
    n = Q.shape[0]
    best_bits = None
    best_value = float("inf")

    for bits in product([0, 1], repeat=n):
        x = np.array(bits, dtype=int)
        value = qubo_value(Q, x)
        if value < best_value:
            best_value = value
            best_bits = x.copy()

    return best_bits, best_value


def qubo_to_ising(Q: np.ndarray):
    """
    Convert QUBO objective x^T Q x with x in {0,1}
    to Ising Hamiltonian H(z) with z in {-1,+1}.
    Returns SparsePauliOp and constant offset.
    """
    n = Q.shape[0]

    linear_z = np.zeros(n, dtype=float)
    quadratic_zz = np.zeros((n, n), dtype=float)
    const = 0.0

    # Diagonal terms: Q_ii x_i
    for i in range(n):
        qii = Q[i, i]
        const += qii / 2.0
        linear_z[i] += -qii / 2.0

    # Off-diagonal terms: for symmetric Q, x^TQx counts both Qij and Qji
    # We only use upper triangle with effective coefficient Qij + Qji
    for i in range(n):
        for j in range(i + 1, n):
            qeff = Q[i, j] + Q[j, i]
            if abs(qeff) < 1e-15:
                continue
            const += qeff / 4.0
            linear_z[i] += -qeff / 4.0
            linear_z[j] += -qeff / 4.0
            quadratic_zz[i, j] += qeff / 4.0

    pauli_strings = []
    coeffs = []

    for i in range(n):
        if abs(linear_z[i]) > 1e-15:
            label = ["I"] * n
            label[i] = "Z"
            pauli_strings.append("".join(label))
            coeffs.append(linear_z[i])

    for i in range(n):
        for j in range(i + 1, n):
            if abs(quadratic_zz[i, j]) > 1e-15:
                label = ["I"] * n
                label[i] = "Z"
                label[j] = "Z"
                pauli_strings.append("".join(label))
                coeffs.append(quadratic_zz[i, j])

    if not pauli_strings:
        pauli_strings = ["I" * n]
        coeffs = [0.0]

    hamiltonian = SparsePauliOp(pauli_strings, coeffs)
    return hamiltonian, const


def apply_cost_layer(qc: QuantumCircuit, gamma: float, hamiltonian: SparsePauliOp):
    """
    Apply exp(-i * gamma * H_C) for diagonal Z / ZZ Hamiltonian.
    """
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        label = pauli.to_label()
        label = label[::-1]  # align with qiskit qubit indexing

        qubits = [i for i, p in enumerate(label) if p == "Z"]
        angle = 2.0 * gamma * float(np.real(coeff))

        if len(qubits) == 0:
            continue
        elif len(qubits) == 1:
            qc.rz(angle, qubits[0])
        elif len(qubits) == 2:
            q0, q1 = qubits
            qc.cx(q0, q1)
            qc.rz(angle, q1)
            qc.cx(q0, q1)
        else:
            raise ValueError(f"Unsupported Pauli term with >2 Z operators: {label}")


def build_qaoa_circuit(hamiltonian: SparsePauliOp, gammas: np.ndarray, betas: np.ndarray):
    n_qubits = hamiltonian.num_qubits
    p = len(gammas)

    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))

    for layer in range(p):
        apply_cost_layer(qc, gammas[layer], hamiltonian)
        for q in range(n_qubits):
            qc.rx(2.0 * betas[layer], q)

    return qc


def qaoa_manual(hamiltonian: SparsePauliOp, const: float, p: int = 2, maxiter: int = 80, seed: int = 42):
    rng = np.random.default_rng(seed)
    estimator = Estimator()

    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        qc = build_qaoa_circuit(hamiltonian, gammas, betas)
        job = estimator.run([qc], [hamiltonian])
        result = job.result()
        energy = float(np.real(result.values[0])) + const
        return energy

    initial_params = np.concatenate(
        [
            rng.uniform(0.0, np.pi, size=p),
            rng.uniform(0.0, np.pi, size=p),
        ]
    )

    res = minimize(
        objective,
        initial_params,
        method="COBYLA",
        options={"maxiter": maxiter},
    )
    return float(res.fun), np.array(res.x, dtype=float)


def bitstring_probabilities(qc: QuantumCircuit):
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities_dict()

    cleaned = {}
    for bitstring, prob in probs.items():
        cleaned[str(bitstring)] = float(prob)

    return dict(sorted(cleaned.items(), key=lambda kv: kv[1], reverse=True))


def bitstring_to_array(bitstring: str) -> np.ndarray:
    return np.array([int(b) for b in bitstring], dtype=int)


def top_counts_from_probabilities(prob_dict, shots: int = 4096, top_k: int = 10):
    items = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    counts = []
    for bitstring, prob in items:
        counts.append((bitstring, int(round(prob * shots))))
    return counts


def save_outputs(
    optimal_energy: float,
    optimal_params: np.ndarray,
    best_bitstring: str,
    best_decision: np.ndarray,
    selected_wells: list,
    selected_qubo_value: float,
    brute_force_bits: np.ndarray,
    brute_force_value: float,
    top_counts: list,
    metadata: dict,
):
    csv_path = OUTPUT_DIR / "qaoa_final_result.csv"
    txt_path = OUTPUT_DIR / "qaoa_engineering_summary.txt"

    result_df = pd.DataFrame(
        [
            {
                "source": metadata.get("source", ""),
                "n_wells": len(metadata["well_names"]),
                "well_names": ", ".join(metadata["well_names"]),
                "best_bitstring": best_bitstring,
                "selected_wells": ", ".join(selected_wells),
                "qaoa_energy": optimal_energy,
                "selected_qubo_value": selected_qubo_value,
                "bruteforce_bitstring": "".join(map(str, brute_force_bits.tolist())),
                "bruteforce_qubo_value": brute_force_value,
                "optimal_parameters": " ".join(f"{x:.6f}" for x in optimal_params),
            }
        ]
    )
    result_df.to_csv(csv_path, index=False)

    lines = []
    lines.append("QAOA Engineering Summary")
    lines.append("=" * 40)
    lines.append(f"Source: {metadata.get('source', '')}")
    lines.append(f"Number of wells: {len(metadata['well_names'])}")
    lines.append(f"Well names: {metadata['well_names']}")
    lines.append("")
    lines.append(f"Best bitstring from QAOA state: {best_bitstring}")
    lines.append(f"Binary decision (0/1): {best_decision.tolist()}")
    lines.append(f"Selected wells: {selected_wells}")
    lines.append(f"QAOA energy: {optimal_energy:.6f}")
    lines.append(f"QUBO value of selected solution: {selected_qubo_value:.6f}")
    lines.append("")
    lines.append(f"Best brute-force bitstring: {''.join(map(str, brute_force_bits.tolist()))}")
    lines.append(f"Best brute-force QUBO value: {brute_force_value:.6f}")
    lines.append("")
    lines.append("Top measurement-like counts:")
    for bitstring, count in top_counts:
        lines.append(f"  {bitstring}: {count}")
    lines.append("")
    lines.append("Optimal parameters:")
    lines.append("  " + " ".join(f"{x:.6f}" for x in optimal_params))

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    return csv_path, txt_path


def main():
    print("\n========== FINAL RESULTS ==========\n")

    Q, metadata = load_volve_qubo(INPUT_FILE)
    well_names = metadata["well_names"]

    print("Loaded Volve QUBO input")
    print("Wells:", well_names)
    print("QUBO Matrix:")
    print(Q)
    print()

    hamiltonian, const = qubo_to_ising(Q)
    print(f"Ising Hamiltonian (offset = {const:.6f}):")
    print(hamiltonian)
    print()

    print("Running QAOA optimization (p=2, COBYLA)...")
    optimal_energy, optimal_params = qaoa_manual(
        hamiltonian=hamiltonian,
        const=const,
        p=2,
        maxiter=80,
        seed=42,
    )

    gammas = optimal_params[:2]
    betas = optimal_params[2:]
    final_qc = build_qaoa_circuit(hamiltonian, gammas, betas)

    prob_dict = bitstring_probabilities(final_qc)
    best_bitstring = next(iter(prob_dict.keys()))
    best_decision = bitstring_to_array(best_bitstring)
    selected_wells = [well_names[i] for i, bit in enumerate(best_decision) if bit == 1]
    selected_qubo_value = qubo_value(Q, best_decision)

    brute_force_bits, brute_force_value = brute_force_best_bitstring(Q)
    top_counts = top_counts_from_probabilities(prob_dict, shots=4096, top_k=10)

    print(f"Optimal Energy from optimizer: {optimal_energy:.6f}")
    print(f"Optimal Angles (gamma, beta): {optimal_params}")
    print()
    print(f"Best Bitstring: {best_bitstring}")
    print(f"Binary Decision (0/1): {best_decision.tolist()}")
    print(f"Selected Wells: {selected_wells}")
    print(f"QUBO Value of Selected Solution: {selected_qubo_value:.6f}")
    print()
    print(f"Brute-force Best Bitstring: {''.join(map(str, brute_force_bits.tolist()))}")
    print(f"Brute-force Best QUBO Value: {brute_force_value:.6f}")
    print()
    print("Top Measurement Counts:")
    for bitstring, count in top_counts[:5]:
        print(f"{bitstring}: {count}")

    csv_path, txt_path = save_outputs(
        optimal_energy=optimal_energy,
        optimal_params=optimal_params,
        best_bitstring=best_bitstring,
        best_decision=best_decision,
        selected_wells=selected_wells,
        selected_qubo_value=selected_qubo_value,
        brute_force_bits=brute_force_bits,
        brute_force_value=brute_force_value,
        top_counts=top_counts,
        metadata=metadata,
    )

    print()
    print(f"Saved CSV summary to: {csv_path}")
    print(f"Saved TXT summary to: {txt_path}")
    print("\n===================================\n")


if __name__ == "__main__":
    main()