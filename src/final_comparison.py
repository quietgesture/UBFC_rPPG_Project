from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
PLOTS_DIR = PROJECT_ROOT / "plots"

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INPUT_FILE = METRICS_DIR / "overall_statistics.csv"


# ==========================================
# LOAD OVERALL STATISTICS
# ==========================================

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("          FINAL POS vs CHROM COMPARISON")
print("=" * 60)

print("\nOverall statistics:")
print(df.to_string(index=False))


# ==========================================
# EXTRACT MEANS
# ==========================================

mae_pos = df.loc[
    df["Metric"] == "MAE (BPM)",
    "POS Mean"
].iloc[0]

mae_chrom = df.loc[
    df["Metric"] == "MAE (BPM)",
    "CHROM Mean"
].iloc[0]

rmse_pos = df.loc[
    df["Metric"] == "RMSE (BPM)",
    "POS Mean"
].iloc[0]

rmse_chrom = df.loc[
    df["Metric"] == "RMSE (BPM)",
    "CHROM Mean"
].iloc[0]

r_pos = df.loc[
    df["Metric"] == "Correlation (R)",
    "POS Mean"
].iloc[0]

r_chrom = df.loc[
    df["Metric"] == "Correlation (R)",
    "CHROM Mean"
].iloc[0]


# ==========================================
# CALCULATE DIFFERENCES
# ==========================================

mae_difference = mae_pos - mae_chrom
rmse_difference = rmse_pos - rmse_chrom
r_difference = r_chrom - r_pos


mae_improvement = (
    (mae_pos - mae_chrom)
    / mae_pos
) * 100

rmse_improvement = (
    (rmse_pos - rmse_chrom)
    / rmse_pos
) * 100


# ==========================================
# PRINT COMPARISON
# ==========================================

print("\n" + "=" * 60)
print("             COMPARISON SUMMARY")
print("=" * 60)

print(f"\nMean MAE:")
print(f"POS   : {mae_pos:.4f} BPM")
print(f"CHROM : {mae_chrom:.4f} BPM")
print(f"Difference: {mae_difference:.4f} BPM")
print(f"Relative difference: {mae_improvement:.2f}%")

print(f"\nMean RMSE:")
print(f"POS   : {rmse_pos:.4f} BPM")
print(f"CHROM : {rmse_chrom:.4f} BPM")
print(f"Difference: {rmse_difference:.4f} BPM")
print(f"Relative difference: {rmse_improvement:.2f}%")

print(f"\nMean Correlation:")
print(f"POS   : {r_pos:.4f}")
print(f"CHROM : {r_chrom:.4f}")
print(f"Difference: {r_difference:.4f}")


# ==========================================
# PLOT 1 — MAE COMPARISON
# ==========================================

methods = ["POS", "CHROM"]
mae_values = [mae_pos, mae_chrom]

plt.figure(figsize=(8, 6))

plt.bar(
    methods,
    mae_values
)

plt.ylabel("Mean MAE (BPM)")
plt.title("Overall Mean MAE: POS vs CHROM")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "overall_mae_comparison.png",
    dpi=300
)

plt.show()


# ==========================================
# PLOT 2 — RMSE COMPARISON
# ==========================================

rmse_values = [
    rmse_pos,
    rmse_chrom
]

plt.figure(figsize=(8, 6))

plt.bar(
    methods,
    rmse_values
)

plt.ylabel("Mean RMSE (BPM)")
plt.title("Overall Mean RMSE: POS vs CHROM")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "overall_rmse_comparison.png",
    dpi=300
)

plt.show()


# ==========================================
# PLOT 3 — CORRELATION COMPARISON
# ==========================================

r_values = [
    r_pos,
    r_chrom
]

plt.figure(figsize=(8, 6))

plt.bar(
    methods,
    r_values
)

plt.ylabel("Mean Correlation (R)")
plt.title("Overall Mean Correlation: POS vs CHROM")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "overall_correlation_comparison.png",
    dpi=300
)

plt.show()


# ==========================================
# SAVE SUMMARY
# ==========================================

summary_file = (
    METRICS_DIR /
    "final_pos_vs_chrom_comparison.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FINAL POS vs CHROM COMPARISON\n"
    )

    f.write(
        "====================================\n\n"
    )

    f.write(
        f"Number of subjects: 42\n\n"
    )

    f.write(
        "Mean MAE (BPM)\n"
    )

    f.write(
        f"POS   : {mae_pos:.4f}\n"
    )

    f.write(
        f"CHROM : {mae_chrom:.4f}\n"
    )

    f.write(
        f"Difference: {mae_difference:.4f}\n"
    )

    f.write(
        f"Relative difference: {mae_improvement:.2f}%\n\n"
    )

    f.write(
        "Mean RMSE (BPM)\n"
    )

    f.write(
        f"POS   : {rmse_pos:.4f}\n"
    )

    f.write(
        f"CHROM : {rmse_chrom:.4f}\n"
    )

    f.write(
        f"Difference: {rmse_difference:.4f}\n"
    )

    f.write(
        f"Relative difference: {rmse_improvement:.2f}%\n\n"
    )

    f.write(
        "Mean Correlation (R)\n"
    )

    f.write(
        f"POS   : {r_pos:.4f}\n"
    )

    f.write(
        f"CHROM : {r_chrom:.4f}\n"
    )

    f.write(
        f"Difference: {r_difference:.4f}\n"
    )


# ==========================================
# COMPLETE
# ==========================================

print("\n" + "=" * 60)
print("       FINAL COMPARISON COMPLETE")
print("=" * 60)

print("\nPlots saved in:")
print(PLOTS_DIR)

print("\nSummary saved to:")
print(summary_file)