# Scalable Hybrid Quantum–Classical Optimization for Oil Reservoir Well Placement (QUBO + QAOA)

## 📌 Overview

This project presents a **scalable hybrid quantum–classical optimization framework** for solving the oil reservoir well placement problem using **Quadratic Unconstrained Binary Optimization (QUBO)** and the **Quantum Approximate Optimization Algorithm (QAOA)**.

The approach transforms reservoir decision-making into a combinatorial optimization problem and leverages quantum-inspired techniques to efficiently explore the solution space.

The framework is validated using a realistic dataset inspired by the **Volve oil field** and benchmarked against classical optimization methods.

---

## 🎯 Key Contributions

* Formulation of oil well placement as a **QUBO problem**
* Application of **QAOA for reservoir optimization**
* Integration of **hybrid quantum–classical workflow**
* Benchmarking against:

  * Exact brute-force solution
  * Simulated Annealing (SA)
* Demonstration of **scalability for real-world field data**

---

## 📊 Case Study (Volve Field)

The model is tested on a dataset inspired by the Volve oil field, incorporating:

* Production rates per well
* Inter-well interference effects
* Spatial and operational constraints

This ensures realistic validation of the optimization framework.

---

## ⚙️ Methodology

### 1️⃣ QUBO Formulation

The well placement problem is expressed as:

* Binary decision variables (0/1)
* Objective: maximize production while minimizing interference
* Encoded into a QUBO matrix

### 2️⃣ Quantum Optimization (QAOA)

* Construction of parameterized quantum circuit
* Optimization of angles (γ, β)
* Measurement-based solution extraction

### 3️⃣ Classical Benchmarking

* Simulated Annealing (SA)
* Exhaustive brute-force search (for validation)

---

## ▶️ How to Run

### 1. Prepare input data

```bash
python prepare_volve_inputs.py
```

### 2. Run QAOA optimization

```bash
python run_qaoa.py
```

### 3. Run classical benchmark (Simulated Annealing)

```bash
python run_sa.py
```

---

## 📈 Results

The framework produces:

* Binary well placement decisions (0/1)
* Optimized QUBO energy values
* Comparison between quantum and classical solutions
* Performance metrics and runtime analysis

  ## 📈 Benchmark Figures

The repository includes benchmark visualizations comparing the proposed hybrid quantum–classical framework against classical methods.

### Objective Energy Comparison
![Objective Energy Comparison](./energy_comparison.png)

### Production Comparison
![Production Comparison](./production_comparison.png)

### Interference Comparison
![Interference Comparison](./interference_comparison.png)

### Runtime Comparison
![Runtime Comparison](./runtime_comparison.png)


### Example Outputs:

* `results/qaoa_final_result.csv`
* `results/sa_final_result.csv`
* `results/volve_qubo_matrix.csv`
* `results_plot.png`

---

## 📊 Visualization

The repository includes:

* Production comparison plots
* Runtime comparison graphs
* Interference visualization
* Optimization performance charts

---

## 🔬 Research Paper

The full research manuscript is included in this repository:

📄 **[Download Full Paper](./paper.pdf)**

---

## 🧠 Technical Stack

* Python
* Qiskit (Quantum simulation)
* NumPy / SciPy
* Matplotlib
* Classical optimization methods

---

## 🔑 Keywords

QUBO, QAOA, Oil Reservoir Optimization, Well Placement, Quantum Computing, Hybrid Optimization, Volve Field, Combinatorial Optimization

---

## 📂 Repository Structure

```
.
├── data/                  # Input datasets
├── results/               # Output results and matrices
├── src/                   # Core algorithms
├── run_qaoa.py            # Quantum optimization
├── run_sa.py              # Classical solver
├── prepare_volve_inputs.py
├── main.tex               # LaTeX manuscript
├── paper.pdf              # Final paper
└── README.md
```

---

## 📜 License

This project is licensed under the MIT License.

---

## 👤 Author

**Tareq Mageed**
Petroleum Engineer – Ministry of Oil
Iraq

---

## 🚀 Future Work

* Extension to multi-objective reservoir optimization
* Integration with real-time field data
* Deployment on actual quantum hardware
* Scaling to larger reservoir models

---

## ⭐ Citation

If you use this work, please cite:

```
Tareq Mageed,
Scalable Hybrid Quantum–Classical Optimization for Oil Reservoir Well Placement,
2026.
```

---

## 🤝 Acknowledgment

This work bridges petroleum engineering and quantum computing, aiming to open new directions in energy optimization using advanced computational paradigms.

---
