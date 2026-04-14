import numpy as np

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.minimum_eigensolvers import QAOA
from qiskit_optimization.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler


def qaoa_solver(Q: np.ndarray, reps: int = 2, maxiter: int = 100):
    """
    Solve a QUBO x^T Q x using QAOA through Qiskit Optimization.
    Returns the same dict shape as the classical solvers.
    """
    n = Q.shape[0]

    problem = QuadraticProgram()
    for i in range(n):
        problem.binary_var(name=f"x{i}")

    linear = {f"x{i}": float(Q[i, i]) for i in range(n)}

    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            # For QuadraticProgram, x_i x_j is represented once.
            # Since our Q is symmetric and energy uses x^T Q x,
            # we combine Q[i,j] and Q[j,i].
            coeff = float(Q[i, j] + Q[j, i])
            if abs(coeff) > 1e-12:
                quadratic[(f"x{i}", f"x{j}")] = coeff

    problem.minimize(linear=linear, quadratic=quadratic)

    sampler = StatevectorSampler(seed=123)
    optimizer = COBYLA(maxiter=maxiter)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps)

    meo = MinimumEigenOptimizer(qaoa)
    result = meo.solve(problem)

    x = np.array(result.x, dtype=int)
    energy = float(x @ Q @ x)

    return {"x": x, "energy": energy}
