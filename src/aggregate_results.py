from pathlib import Path
import re
import csv


# ==========================================
# PROJECT DIRECTORIES
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

OUTPUT_CSV = METRICS_DIR / "all_subject_metrics.csv"


# ==========================================
# FIND SUBJECTS
# ==========================================

pos_files = sorted(
    METRICS_DIR.glob("subject*_pos_metrics.txt"),
    key=lambda p: int(
        re.search(r"subject(\d+)", p.name).group(1)
    )
)

chrom_files = sorted(
    METRICS_DIR.glob("subject*_chrom_metrics.txt"),
    key=lambda p: int(
        re.search(r"subject(\d+)", p.name).group(1)
    )
)


print("==========================================")
print("       UBFC rPPG RESULTS AGGREGATION")
print("==========================================")

print(f"POS metric files found   : {len(pos_files)}")
print(f"CHROM metric files found : {len(chrom_files)}")


# ==========================================
# FUNCTION TO READ METRICS
# ==========================================

def read_metrics(file_path):

    metrics = {
        "MAE": None,
        "RMSE": None,
        "R": None
    }

    if not file_path.exists():
        return metrics

    text = file_path.read_text(
        encoding="utf-8"
    )

    mae_match = re.search(
        r"MAE\s*:\s*([-+]?\d*\.?\d+)",
        text
    )

    rmse_match = re.search(
        r"RMSE\s*:\s*([-+]?\d*\.?\d+)",
        text
    )

    r_match = re.search(
        r"R\s*:\s*([-+]?\d*\.?\d+)",
        text
    )

    if mae_match:
        metrics["MAE"] = float(
            mae_match.group(1)
        )

    if rmse_match:
        metrics["RMSE"] = float(
            rmse_match.group(1)
        )

    if r_match:
        metrics["R"] = float(
            r_match.group(1)
        )

    return metrics


# ==========================================
# COLLECT SUBJECT NUMBERS
# ==========================================

subjects = set()

for file in pos_files + chrom_files:

    match = re.search(
        r"subject(\d+)",
        file.name
    )

    if match:
        subjects.add(
            int(match.group(1))
        )

subjects = sorted(subjects)


print(f"Unique subjects found   : {len(subjects)}")
print()


# ==========================================
# CREATE LOOKUP DICTIONARIES
# ==========================================

pos_lookup = {}

for file in pos_files:

    match = re.search(
        r"subject(\d+)",
        file.name
    )

    if match:

        subject = int(
            match.group(1)
        )

        pos_lookup[subject] = read_metrics(
            file
        )


chrom_lookup = {}

for file in chrom_files:

    match = re.search(
        r"subject(\d+)",
        file.name
    )

    if match:

        subject = int(
            match.group(1)
        )

        chrom_lookup[subject] = read_metrics(
            file
        )


# ==========================================
# BUILD FINAL TABLE
# ==========================================

rows = []

for subject in subjects:

    pos = pos_lookup.get(
        subject,
        {
            "MAE": None,
            "RMSE": None,
            "R": None
        }
    )

    chrom = chrom_lookup.get(
        subject,
        {
            "MAE": None,
            "RMSE": None,
            "R": None
        }
    )

    row = {
        "Subject": subject,

        "POS_MAE": pos["MAE"],
        "POS_RMSE": pos["RMSE"],
        "POS_R": pos["R"],

        "CHROM_MAE": chrom["MAE"],
        "CHROM_RMSE": chrom["RMSE"],
        "CHROM_R": chrom["R"]
    }

    rows.append(row)


# ==========================================
# SAVE CSV
# ==========================================

fieldnames = [
    "Subject",
    "POS_MAE",
    "POS_RMSE",
    "POS_R",
    "CHROM_MAE",
    "CHROM_RMSE",
    "CHROM_R"
]

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(rows)


# ==========================================
# PRINT RESULTS
# ==========================================

print("==========================================")
print("           AGGREGATION COMPLETE")
print("==========================================")

print()

for row in rows:

    print(
        f"Subject {row['Subject']:>2} | "
        f"POS MAE: {row['POS_MAE']} | "
        f"POS RMSE: {row['POS_RMSE']} | "
        f"POS R: {row['POS_R']} | "
        f"CHROM MAE: {row['CHROM_MAE']} | "
        f"CHROM RMSE: {row['CHROM_RMSE']} | "
        f"CHROM R: {row['CHROM_R']}"
    )


print()
print("Saved final table to:")
print(OUTPUT_CSV)