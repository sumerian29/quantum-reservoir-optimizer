from pathlib import Path
import csv


PROJECT_ROOT = Path(".").resolve()


def find_first_existing(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def format_float(value, decimals=4):
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


def load_summary_csv(csv_path: Path):
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def build_latex_table(rows):
    methods_order = ["Exact", "QAOA", "Random", "SA"]

    def sort_key(row):
        method_rank = methods_order.index(row["method"]) if row["method"] in methods_order else 999
        return (int(row["n_wells"]), method_rank)

    rows = sorted(rows, key=sort_key)

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comparison of optimization methods across different candidate well counts.}")
    lines.append(r"\label{tab:benchmark_comparison}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Wells & Energy Mean & Production Mean & Interference Mean & Runtime Mean (s) & Runtime Std \\")
    lines.append(r"\midrule")

    for row in rows:
        method = row["method"]
        n_wells = row["n_wells"]
        energy_mean = format_float(row["energy_mean"], 4)
        production_mean = format_float(row["production_mean"], 4)
        interference_mean = format_float(row["interference_mean"], 4)
        runtime_mean = format_float(row["runtime_mean"], 6)
        runtime_std = format_float(row["runtime_std"], 6)

        lines.append(
            f"{method} & {n_wells} & {energy_mean} & {production_mean} & "
            f"{interference_mean} & {runtime_mean} & {runtime_std} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def build_figures_block(figure_paths):
    def latex_path(path: Path):
        return path.as_posix()

    energy = latex_path(figure_paths["energy"])
    production = latex_path(figure_paths["production"])
    interference = latex_path(figure_paths["interference"])
    runtime = latex_path(figure_paths["runtime"])

    lines = []
    lines.append("% =============================")
    lines.append("% Figures block")
    lines.append("% =============================")
    lines.append(r"\begin{figure}[H]")
    lines.append(r"\centering")
    lines.append(rf"\includegraphics[width=0.8\textwidth]{{{energy}}}")
    lines.append(r"\caption{Objective energy comparison across optimization methods. Lower values are better.}")
    lines.append(r"\label{fig:energy_comparison}")
    lines.append(r"\end{figure}")
    lines.append("")

    lines.append(r"\begin{figure}[H]")
    lines.append(r"\centering")
    lines.append(rf"\includegraphics[width=0.8\textwidth]{{{production}}}")
    lines.append(r"\caption{Production comparison across optimization methods. Higher values are better.}")
    lines.append(r"\label{fig:production_comparison}")
    lines.append(r"\end{figure}")
    lines.append("")

    lines.append(r"\begin{figure}[H]")
    lines.append(r"\centering")
    lines.append(rf"\includegraphics[width=0.8\textwidth]{{{interference}}}")
    lines.append(r"\caption{Interference penalty comparison across optimization methods. Lower values are better.}")
    lines.append(r"\label{fig:interference_comparison}")
    lines.append(r"\end{figure}")
    lines.append("")

    lines.append(r"\begin{figure}[H]")
    lines.append(r"\centering")
    lines.append(rf"\includegraphics[width=0.8\textwidth]{{{runtime}}}")
    lines.append(r"\caption{Runtime comparison across optimization methods. Lower values are better.}")
    lines.append(r"\label{fig:runtime_comparison}")
    lines.append(r"\end{figure}")

    return "\n".join(lines)


def main():
    # Detect summary CSV automatically
    summary_csv_candidates = [
        PROJECT_ROOT / "comparison_summary.csv",
        PROJECT_ROOT / "results" / "comparison_summary.csv",
    ]
    summary_csv_path = find_first_existing(summary_csv_candidates)

    if summary_csv_path is None:
        raise FileNotFoundError(
            "Could not find comparison_summary.csv in either the project root or results/ folder."
        )

    # Detect figures automatically
    figure_candidates = {
        "energy": [
            PROJECT_ROOT / "energy_comparison.png",
            PROJECT_ROOT / "results" / "energy_comparison.png",
        ],
        "production": [
            PROJECT_ROOT / "production_comparison.png",
            PROJECT_ROOT / "results" / "production_comparison.png",
        ],
        "interference": [
            PROJECT_ROOT / "interference_comparison.png",
            PROJECT_ROOT / "results" / "interference_comparison.png",
        ],
        "runtime": [
            PROJECT_ROOT / "runtime_comparison.png",
            PROJECT_ROOT / "results" / "runtime_comparison.png",
        ],
    }

    figure_paths = {}
    missing = []

    for key, candidates in figure_candidates.items():
        found = find_first_existing(candidates)
        if found is None:
            missing.append(key)
        else:
            figure_paths[key] = found.relative_to(PROJECT_ROOT)

    if missing:
        raise FileNotFoundError(
            "Missing required figure files for: " + ", ".join(missing)
        )

    # Load CSV
    rows = load_summary_csv(summary_csv_path)
    if not rows:
        raise ValueError("comparison_summary.csv was found, but it is empty.")

    # Build outputs
    latex_table = build_latex_table(rows)
    figures_block = build_figures_block(figure_paths)

    # Write output files
    table_out = PROJECT_ROOT / "table1.tex"
    figures_out = PROJECT_ROOT / "figures_block.tex"
    report_out = PROJECT_ROOT / "latex_assets_report.txt"

    table_out.write_text(latex_table, encoding="utf-8")
    figures_out.write_text(figures_block, encoding="utf-8")

    report_lines = [
        "LaTeX Assets Report",
        "===================",
        "",
        f"Summary CSV used: {summary_csv_path.relative_to(PROJECT_ROOT).as_posix()}",
        "",
        "Figures used:",
        f"  Energy:        {figure_paths['energy'].as_posix()}",
        f"  Production:    {figure_paths['production'].as_posix()}",
        f"  Interference:  {figure_paths['interference'].as_posix()}",
        f"  Runtime:       {figure_paths['runtime'].as_posix()}",
        "",
        "Generated files:",
        f"  {table_out.name}",
        f"  {figures_out.name}",
        f"  {report_out.name}",
        "",
        "How to use in LaTeX:",
        "  1) Put \\usepackage{booktabs} and \\usepackage{float} in the preamble.",
        "  2) Insert the table using: \\input{table1.tex}",
        "  3) Insert the figures using: \\input{figures_block.tex}",
    ]
    report_out.write_text("\n".join(report_lines), encoding="utf-8")

    print("Done.")
    print(f"Summary CSV used: {summary_csv_path.relative_to(PROJECT_ROOT).as_posix()}")
    print("Figures used:")
    print(f"  Energy:        {figure_paths['energy'].as_posix()}")
    print(f"  Production:    {figure_paths['production'].as_posix()}")
    print(f"  Interference:  {figure_paths['interference'].as_posix()}")
    print(f"  Runtime:       {figure_paths['runtime'].as_posix()}")
    print("")
    print("Generated:")
    print(f"  {table_out.name}")
    print(f"  {figures_out.name}")
    print(f"  {report_out.name}")


if __name__ == "__main__":
    main()