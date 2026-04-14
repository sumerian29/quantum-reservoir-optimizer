import numpy as np

def qaoa_solver(Q):
    """
    Simple placeholder for QAOA solver
    (Simulated using random binary optimization)

    Parameters:
        Q (np.ndarray): QUBO matrix

    Returns:
        dict: solution dictionary with 'x' and 'energy'
    """

    n = Q.shape[0]

    # Generate random binary solution
    x = np.random.randint(0, 2, size=n)

    # Compute QUBO energy: x^T Q x
    energy = x @ Q @ x

    return {
        "x": x,
        "energy": energy
    }
