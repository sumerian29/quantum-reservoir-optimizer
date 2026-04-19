# Scalable Hybrid Quantum–Classical Optimization for Oil Reservoir Well Placement (QUBO + QAOA)

## Overview

This project presents a hybrid quantum–classical optimization framework for solving the oil reservoir well placement problem using:

- Quadratic Unconstrained Binary Optimization (QUBO)
- Quantum Approximate Optimization Algorithm (QAOA)

The approach transforms reservoir decision-making into a combinatorial optimization problem and leverages quantum-inspired techniques to explore the solution space.

---

## Key Contributions

- Formulation of well placement as a QUBO problem
- Implementation of QAOA for reservoir optimization
- Hybrid quantum–classical workflow
- Benchmarking against:
  - Brute Force (exact solution)
  - Simulated Annealing (SA)

---

## Methodology

### 1. QUBO Formulation
- Binary decision variables (0/1)
- Objective:
  - Maximize production
  - Minimize inter-well interference

### 2. Quantum Optimization (QAOA)
- Parameterized quantum circuit
- Optimization of angles (γ, β)
- Measurement-based solution extraction

### 3. Classical Benchmarking
- Simulated Annealing (SA)
- Brute-force enumeration

---

## Experimental Setup

- Number of wells: **N = 4**
- Problem type: combinatorial optimization (proof-of-concept)
- Objective: energy minimization

---

## Results

| Method        | Energy     | Approximation Ratio |
|--------------|------------|--------------------|
| Brute Force  | -1.7025    | 1.00               |
| SA           | -1.7025    | 1.00               |
| QAOA         | -1.7025    | 1.00               |

✔ QAOA successfully reached the optimal solution in this instance.

---

## Generated Figures

All figures are located in:
