from pathlib import Path

import numpy as np
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


# ==========================================
# LOAD AGGREGATED RESULTS
# ==========================================

csv_path = METRICS_DIR / "all_subject_metrics.csv"

data = np.genfromtxt(
    csv_path,
    delimiter=",",
    names=True
)

subjects = data["Subject"]

pos_mae = data["POS_MAE"]
chrom_mae = data["CHROM_MAE"]

pos_rmse = data["POS_RMSE"]
chrom_rmse = data["CHROM_RMSE"]

pos_r = data["POS_R"]
chrom_r = data["CHROM_R"]


print("==========================================")
print("         RESULT VISUALIZATION")
print("==========================================")

print(f"Subjects plotted: {len(subjects)}")


# ==========================================
# PLOT 1 — MAE
# ==========================================

plt.figure(figsize=(14, 7))

plt.plot(
    subjects,
    pos_mae,
    marker="o",
    label="POS"
)

plt.plot(
    subjects,
    chrom_mae,
    marker="o",
    label="CHROM"
)

plt.xlabel("Subject")
plt.ylabel("MAE (BPM)")

plt.title(
    "Subject-wise MAE: POS vs CHROM"
)

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "mae_pos_vs_chrom.png",
    dpi=300
)

plt.show()


# ==========================================
# PLOT 2 — RMSE
# ==========================================

plt.figure(figsize=(14, 7))

plt.plot(
    subjects,
    pos_rmse,
    marker="o",
    label="POS"
)

plt.plot(
    subjects,
    chrom_rmse,
    marker="o",
    label="CHROM"
)

plt.xlabel("Subject")
plt.ylabel("RMSE (BPM)")

plt.title(
    "Subject-wise RMSE: POS vs CHROM"
)

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "rmse_pos_vs_chrom.png",
    dpi=300
)

plt.show()


# ==========================================
# PLOT 3 — CORRELATION
# ==========================================

plt.figure(figsize=(14, 7))

plt.plot(
    subjects,
    pos_r,
    marker="o",
    label="POS"
)

plt.plot(
    subjects,
    chrom_r,
    marker="o",
    label="CHROM"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Subject")
plt.ylabel("Correlation (R)")

plt.title(
    "Subject-wise Correlation: POS vs CHROM"
)

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "correlation_pos_vs_chrom.png",
    dpi=300
)

plt.show()


# ==========================================
# COMPLETE
# ==========================================

print()
print("Plots saved successfully!")

print(
    PLOTS_DIR / "mae_pos_vs_chrom.png"
)

print(
    PLOTS_DIR / "rmse_pos_vs_chrom.png"
)

print(
    PLOTS_DIR / "correlation_pos_vs_chrom.png"
)