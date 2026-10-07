"""
Overall Statistical Analysis
UBFC rPPG Project

Reads:
    results/metrics/all_subject_metrics.csv

Calculates overall statistics for:
    POS
    CHROM

Metrics:
    MAE
    RMSE
    Correlation (R)

Outputs:
    results/metrics/overall_statistics.csv
    results/metrics/overall_statistics.txt
"""

import os
import pandas as pd
import numpy as np

from config import RESULTS_DIR


# ============================================================
# PATHS
# ============================================================

METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

INPUT_FILE = os.path.join(
    METRICS_DIR,
    "all_subject_metrics.csv"
)

OUTPUT_CSV = os.path.join(
    METRICS_DIR,
    "overall_statistics.csv"
)

OUTPUT_TXT = os.path.join(
    METRICS_DIR,
    "overall_statistics.txt"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("       UBFC rPPG OVERALL STATISTICAL ANALYSIS")
print("=" * 60)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Could not find:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"\nInput file: {INPUT_FILE}")
print(f"Subjects found: {len(df)}")


# ============================================================
# EXPECTED COLUMNS
# ============================================================

required_columns = [
    "Subject",
    "POS_MAE",
    "POS_RMSE",
    "POS_R",
    "CHROM_MAE",
    "CHROM_RMSE",
    "CHROM_R"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    print("\nAvailable columns:")
    print(df.columns.tolist())

    raise ValueError(
        f"\nMissing required columns: {missing}"
    )


# ============================================================
# STATISTICAL FUNCTION
# ============================================================

def calculate_statistics(series):
    """
    Calculate descriptive statistics for one metric.
    """

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    return {
        "Mean": series.mean(),
        "Median": series.median(),
        "Std": series.std(ddof=1),
        "Minimum": series.min(),
        "Maximum": series.max()
    }


# ============================================================
# CALCULATE POS STATISTICS
# ============================================================

pos_mae = calculate_statistics(df["POS_MAE"])
pos_rmse = calculate_statistics(df["POS_RMSE"])
pos_r = calculate_statistics(df["POS_R"])


# ============================================================
# CALCULATE CHROM STATISTICS
# ============================================================

chrom_mae = calculate_statistics(df["CHROM_MAE"])
chrom_rmse = calculate_statistics(df["CHROM_RMSE"])
chrom_r = calculate_statistics(df["CHROM_R"])


# ============================================================
# CREATE SUMMARY TABLE
# ============================================================

statistics_table = pd.DataFrame(
    {
        "Metric": [
            "MAE (BPM)",
            "RMSE (BPM)",
            "Correlation (R)"
        ],

        "POS Mean": [
            pos_mae["Mean"],
            pos_rmse["Mean"],
            pos_r["Mean"]
        ],

        "POS Median": [
            pos_mae["Median"],
            pos_rmse["Median"],
            pos_r["Median"]
        ],

        "POS Std": [
            pos_mae["Std"],
            pos_rmse["Std"],
            pos_r["Std"]
        ],

        "POS Minimum": [
            pos_mae["Minimum"],
            pos_rmse["Minimum"],
            pos_r["Minimum"]
        ],

        "POS Maximum": [
            pos_mae["Maximum"],
            pos_rmse["Maximum"],
            pos_r["Maximum"]
        ],

        "CHROM Mean": [
            chrom_mae["Mean"],
            chrom_rmse["Mean"],
            chrom_r["Mean"]
        ],

        "CHROM Median": [
            chrom_mae["Median"],
            chrom_rmse["Median"],
            chrom_r["Median"]
        ],

        "CHROM Std": [
            chrom_mae["Std"],
            chrom_rmse["Std"],
            chrom_r["Std"]
        ],

        "CHROM Minimum": [
            chrom_mae["Minimum"],
            chrom_rmse["Minimum"],
            chrom_r["Minimum"]
        ],

        "CHROM Maximum": [
            chrom_mae["Maximum"],
            chrom_rmse["Maximum"],
            chrom_r["Maximum"]
        ]
    }
)


# ============================================================
# ROUND VALUES
# ============================================================

statistics_table = statistics_table.round(4)


# ============================================================
# SAVE CSV
# ============================================================

statistics_table.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("              OVERALL STATISTICS")
print("=" * 60)

print("\n")
print(statistics_table.to_string(index=False))


# ============================================================
# SAVE TEXT REPORT
# ============================================================

with open(
    OUTPUT_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "============================================================\n"
    )

    f.write(
        "       UBFC rPPG OVERALL STATISTICAL ANALYSIS\n"
    )

    f.write(
        "============================================================\n\n"
    )

    f.write(
        f"Number of subjects: {len(df)}\n\n"
    )

    f.write(
        statistics_table.to_string(index=False)
    )

    f.write("\n\n")

    f.write(
        "Metric definitions:\n"
    )

    f.write(
        "MAE  = Mean Absolute Error (BPM)\n"
    )

    f.write(
        "RMSE = Root Mean Square Error (BPM)\n"
    )

    f.write(
        "R    = Pearson correlation coefficient\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("       STATISTICAL ANALYSIS COMPLETE")
print("=" * 60)

print("\nSaved overall statistics to:")
print(OUTPUT_CSV)

print("\nSaved text report to:")
print(OUTPUT_TXT)

print("\nDone.")