from pathlib import Path
import random
import numpy as np
import pandas as pd


INPUT_FILE = Path("results") / "volve_qubo_inputs.npz"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_volve_qubo(npz_path: Path):
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing input file: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    Q = data["Q"].astype(float)
    well_names = data["well_names"].tolist()

    metadata = {
        "well_names": well_names,
        "source": "Volve-inspired field dataset",
    }
    return Q, metadata


def qubo_value(x: np.ndarray, Q: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(x @ Q @ x)


def random_solution(n: int) -> np.ndarray:
    return np.array([random.randint(0, 1) for _ in range(n)], dtype=int)


def neighbor(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    i = random.randint(0, len(x) - 1)
    y[i] = 1 - y[i]
    return y


def simulated_annealing(
    Q: np.ndarray,
    initial_temperature: float = 10.0,
    cooling_rate: float = 0.995,
    iterations: int = 5000,
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)

    n = Q.shape[0]
    x = random_solution(n)
    current_val = qubo_value(x, Q)

    best_x = x.copy()
    best_val = current_val

    T = initial_temperature

    for _ in range(iterations):
        x_new = neighbor(x)
        val_new = qubo_value(x_new, Q)
        delta = val_new - current_val

        if delta < 0:
            x = x_new
            current_val = val_new
        else:
            if random.random() < np.exp(-delta / max(T, 1e-12)):
                x = x_new
                current_val = val_new

        if current_val < best_val:
            best_x = x.copy()
            best_val = current_val

        T *= cooling_rate

    return best_x, float(best_val)


def save_outputs(best_x, best_val, metadata):
    selected_wells = [metadata["well_names"][i] for i, bit in enumerate(best_x) if bit == 1]
    bitstring = "".join(str(int(b)) for b in best_x.tolist())

    csv_path = OUTPUT_DIR / "sa_final_result.csv"
    txt_path = OUTPUT_DIR / "sa_engineering_summary.txt"

    df = pd.DataFrame(
        [
            {
                "source": metadata.get("source", ""),
                "n_wells": len(metadata["well_names"]),
                "well_names": ", ".join(metadata["well_names"]),
                "best_bitstring": bitstring,
                "selected_wells": ", ".join(selected_wells),
                "qubo_value": best_val,
            }
        ]
    )
    df.to_csv(csv_path, index=False)

    lines = []
    lines.append("Simulated Annealing Engineering Summary")
    lines.append("=" * 42)
    lines.append(f"Source: {metadata.get('source', '')}")
    lines.append(f"Number of wells: {len(metadata['well_names'])}")
    lines.append(f"Well names: {metadata['well_names']}")
    lines.append("")
    lines.append(f"Best bitstring: {bitstring}")
    lines.append(f"Binary decision (0/1): {best_x.tolist()}")
    lines.append(f"Selected wells: {selected_wells}")
    lines.append(f"Best QUBO value: {best_val:.6f}")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    return csv_path, txt_path, selected_wells, bitstring


def main():
    print("\n========== SIMULATED ANNEALING RESULTS ==========\n")

    Q, metadata = load_volve_qubo(INPUT_FILE)
    well_names = metadata["well_names"]

    print("Loaded Volve QUBO input")
    print("Wells:", well_names)
    print("QUBO Matrix:")
    print(Q)
    print()

    best_x, best_val = simulated_annealing(
        Q=Q,
        initial_temperature=10.0,
        cooling_rate=0.995,
        iterations=5000,
        seed=42,
    )

    csv_path, txt_path, selected_wells, bitstring = save_outputs(best_x, best_val, metadata)

    print(f"Best Bitstring: {bitstring}")
    print(f"Binary Decision (0/1): {best_x.tolist()}")
    print(f"Selected Wells: {selected_wells}")
    print(f"Best QUBO Value: {best_val:.6f}")
    print()
    print(f"Saved CSV summary to: {csv_path}")
    print(f"Saved TXT summary to: {txt_path}")
    print("\n===============================================\n")


if __name__ == "__main__":
    main()