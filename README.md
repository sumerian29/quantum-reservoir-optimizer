# 🧠 Quantum Reservoir Optimizer

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Status](https://img.shields.io/badge/status-research--grade-orange)

A scalable hybrid quantum-classical framework for **oil reservoir well placement optimization** using **QUBO modeling** and **QAOA**.

---

## 🚀 Overview

This repository presents a research-oriented optimization framework for oil reservoir well placement.

It combines:

- Classical optimization
- Quantum-inspired / quantum optimization
- QUBO-based problem formulation
- Benchmarking and visualization tools

The framework is designed to study trade-offs between:

- **Production maximization**
- **Interference minimization**
- **Runtime efficiency**
- **Scalability**

---

## ⚙️ Core Methods

The framework benchmarks the following methods:

- **Exact Solver (Brute Force)**
- **Simulated Annealing (SA)**
- **Random Search**
- **QAOA (Quantum Approximate Optimization Algorithm)**

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
