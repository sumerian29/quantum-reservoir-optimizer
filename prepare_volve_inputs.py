from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
WELLS_FILE = DATA_DIR / "volve_wells.csv"
PROD_FILE = DATA_DIR / "volve_production.csv"
OUTPUT_DIR = Path("results")

DISTANCE_SCALE = 1000.0
WELL_COST_PENALTY = 0.10
MAX_WELLS = 10


def normalize(series: pd.Series) -> pd.Series:
    smin = series.min()
    smax = series.max()
    if abs(smax - smin) < 1e-12:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - smin) / (smax - smin)


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(coords[i] - coords[j])
    return dist


def compute_interference_matrix(distance_matrix: np.ndarray, scale: float = 1000.0) -> np.ndarray:
    interference = np.exp(-distance_matrix / scale)
    np.fill_diagonal(interference, 0.0)
    return interference


def build_qubo_from_volve(production: np.ndarray, interference: np.ndarray, well_cost_penalty: float = 0.10) -> np.ndarray:
    n = len(production)
    Q = np.zeros((n, n), dtype=float)

    for i in range(n):
        Q[i, i] = -production[i] + well_cost_penalty

    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = interference[i, j]
            Q[j, i] = Q[i, j]

    return Q


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not WELLS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {WELLS_FILE}")

    if not PROD_FILE.exists():
        raise FileNotFoundError(f"Missing file: {PROD_FILE}")

    wells_df = pd.read_csv(WELLS_FILE)
    prod_df = pd.read_csv(PROD_FILE)

    required_well_cols = {"well", "x", "y"}
    required_prod_cols = {"well", "oil_rate"}

    if not required_well_cols.issubset(wells_df.columns):
        raise ValueError(f"{WELLS_FILE.name} must contain columns: {sorted(required_well_cols)}")

    if not required_prod_cols.issubset(prod_df.columns):
        raise ValueError(f"{PROD_FILE.name} must contain columns: {sorted(required_prod_cols)}")

    prod_agg = (
        prod_df.groupby("well", as_index=False)["oil_rate"]
        .mean()
        .rename(columns={"oil_rate": "avg_oil_rate"})
    )

    merged = wells_df.merge(prod_agg, on="well", how="inner")

    if merged.empty:
        raise ValueError("No overlapping wells found between coordinates and production files.")

    merged = merged.head(MAX_WELLS).copy()
    merged["production_norm"] = normalize(merged["avg_oil_rate"])

    coords = merged[["x", "y"]].to_numpy(dtype=float)
    distance_matrix = compute_distance_matrix(coords)
    interference_matrix = compute_interference_matrix(distance_matrix, scale=DISTANCE_SCALE)

    production_vector = merged["production_norm"].to_numpy(dtype=float)
    well_names = merged["well"].tolist()

    Q = build_qubo_from_volve(
        production=production_vector,
        interference=interference_matrix,
        well_cost_penalty=WELL_COST_PENALTY,
    )

    npz_path = OUTPUT_DIR / "volve_qubo_inputs.npz"
    np.savez(
        npz_path,
        Q=Q,
        production=production_vector,
        distance_matrix=distance_matrix,
        interference_matrix=interference_matrix,
        well_names=np.array(well_names, dtype=object),
        x_coords=merged["x"].to_numpy(dtype=float),
        y_coords=merged["y"].to_numpy(dtype=float),
        raw_avg_oil_rate=merged["avg_oil_rate"].to_numpy(dtype=float),
    )

    preview_path = OUTPUT_DIR / "volve_qubo_preview.csv"
    preview_df = merged[["well", "x", "y", "avg_oil_rate", "production_norm"]].copy()
    preview_df.to_csv(preview_path, index=False)

    q_csv_path = OUTPUT_DIR / "volve_qubo_matrix.csv"
    q_df = pd.DataFrame(Q, index=well_names, columns=well_names)
    q_df.to_csv(q_csv_path)

    print("\n========== VOLVE INPUT PREPARATION COMPLETE ==========\n")
    print(f"Wells used: {len(well_names)}")
    print("Well names:", well_names)
    print()
    print("Normalized production vector:")
    print(production_vector)
    print()
    print("Distance matrix shape:", distance_matrix.shape)
    print("Interference matrix shape:", interference_matrix.shape)
    print("QUBO matrix shape:", Q.shape)
    print()
    print("Saved files:")
    print(f"  {npz_path}")
    print(f"  {preview_path}")
    print(f"  {q_csv_path}")
    print("\n======================================================\n")


if __name__ == "__main__":
    main()