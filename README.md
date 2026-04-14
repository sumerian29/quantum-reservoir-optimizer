# 🧠 Quantum Reservoir Optimizer

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10+-blue)

A scalable hybrid quantum-classical framework for oil reservoir well placement optimization using **QUBO modeling** and **QAOA**.

---

## 🚀 Overview

This repository presents a research-oriented optimization framework for oil reservoir well placement.  
It combines:

- Classical optimization
- Quantum-inspired / quantum optimization
- QUBO-based problem formulation
- Benchmarking and visualization tools

The framework is designed to study the trade-off between:

- **Production maximization**
- **Interference minimization**
- **Runtime efficiency**
- **Scalability**

---

## ⚙️ Core Methods

The project currently benchmarks the following methods:

- **Exact Solver**
- **Simulated Annealing (SA)**
- **Random Search**
- **QAOA**

---

## 📂 Repository Structure

```text
quantum-reservoir-optimizer/
│
├── docs/
├── examples/
├── notebooks/
├── results/
├── src/
│   ├── classical/
│   ├── quantum/
│   ├── utils/
│   └── qubo_builder.py
│
├── comparison.py
├── config.yaml
├── main.py
├── requirements.txt
├── README.md
└── LICENSE
