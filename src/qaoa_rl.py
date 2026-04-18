import numpy as np
from qiskit_aer import Aer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.quantum_info import SparsePauliOp

# Try the modern sampler first, then fall back
SAMPLER_SOURCE = None
sampler = None

try:
    from qiskit.primitives import StatevectorSampler
    sampler = StatevectorSampler()
    SAMPLER_SOURCE = "StatevectorSampler"
except Exception:
    try:
        from qiskit.primitives import Sampler
        sampler = Sampler()
        SAMPLER_SOURCE = "Sampler"
    except Exception:
        sampler = None
        SAMPLER_SOURCE = "None"


def build_qubo(production, distances):
    """
    Build a simple physics-inspired QUBO matrix.
    """
    n = len(production)
    Q = np.zeros((n, n), dtype=float)

    # Diagonal: production reward
    for i in range(n):
        Q[i, i] = -production[i]

    # Off-diagonal: distance-decay + simple synergy term
    for i in range(n):
        for j in range(i + 1, n):
            interaction = (
                0.2 * np.exp(-distances[i, j] / 500.0)
                - 0.3
                + 0.1 * (production[i] * production[j] / (distances[i, j] + 1.0))
            )
            Q[i, j] = interaction
            Q[j, i] = interaction

    return Q


def qubo_to_sparse_pauli(Q):
    """
    Convert QUBO matrix into SparsePauliOp using Z and ZZ terms.
    This is a simplified demonstrator, not a full exact QUBO-to-Ising conversion.
    """
    n = Q.shape[0]
    paulis = []
    coeffs = []

    # Single Z terms
    for i in range(n):
        label = ["I"] * n
        label[n - 1 - i] = "Z"
        paulis.append("".join(label))
        coeffs.append(float(Q[i, i]))

    # ZZ terms
    for i in range(n):
        for j in range(i + 1, n):
            if abs(Q[i, j]) > 1e-12:
                label = ["I"] * n
                label[n - 1 - i] = "Z"
                label[n - 1 - j] = "Z"
                paulis.append("".join(label))
                coeffs.append(float(Q[i, j]))

    return SparsePauliOp(paulis, coeffs=coeffs)


def extract_best_bitstring(result, n):
    """
    Try to extract the most likely bitstring from the result object.
    """
    try:
        if hasattr(result, "eigenstate") and result.eigenstate is not None:
            eigenstate = result.eigenstate

            if hasattr(eigenstate, "binary_probabilities"):
                probs = eigenstate.binary_probabilities()
                if probs:
                    best = max(probs, key=probs.get)
                    return best.zfill(n)

            if isinstance(eigenstate, dict) and eigenstate:
                best = max(eigenstate, key=eigenstate.get)
                return str(best).zfill(n)

        return "Unavailable"
    except Exception:
        return "Unavailable"


def main():
    np.random.seed(42)

    # Example small synthetic case
    n = 5
    production = np.array([0.92, 0.75, 0.88, 0.81, 0.69], dtype=float)

    distances = np.array([
        [0.0, 420.0, 680.0, 510.0, 790.0],
        [420.0, 0.0, 390.0, 610.0, 730.0],
        [680.0, 390.0, 0.0, 440.0, 520.0],
        [510.0, 610.0, 440.0, 0.0, 360.0],
        [790.0, 730.0, 520.0, 360.0, 0.0],
    ], dtype=float)

    Q = build_qubo(production, distances)
    operator = qubo_to_sparse_pauli(Q)

    print("=" * 60)
    print("QAOA demo for well placement")
    print("Sampler source:", SAMPLER_SOURCE)
    print("=" * 60)

    optimizer = COBYLA(maxiter=100)

    if sampler is None:
        raise RuntimeError(
            "No compatible sampler found in qiskit.primitives. "
            "Please install/upgrade qiskit and qiskit-algorithms."
        )

    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=2,
    )

    result = qaoa.compute_minimum_eigenvalue(operator)

    print("QAOA run completed successfully")
    print("Best energy:", float(np.real(result.eigenvalue)))
    print("Best bitstring:", extract_best_bitstring(result, n))
    print("Production vector:", production.tolist())
    print("QUBO matrix:")
    print(np.round(Q, 4))


if __name__ == "__main__":
    main()