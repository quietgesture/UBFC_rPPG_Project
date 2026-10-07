import numpy as np
from scipy.stats import pearsonr
import sys

from config import (
    DATASET_ROOT,
    RESULTS_DIR,
    WINDOW_SECONDS,
    get_subject_fps
)


# --------------------------------------------------
# Check command-line argument
# --------------------------------------------------

if len(sys.argv) < 2:
    print(
        "Usage: python src/chrom_evaluation.py <subject_number>"
    )
    print(
        "Example: python src/chrom_evaluation.py 1"
    )
    sys.exit()

subject_number = int(sys.argv[1])


# --------------------------------------------------
# Paths
# --------------------------------------------------

subject_dir = DATASET_ROOT / f"subject{subject_number}"

ground_truth_path = (
    subject_dir / "ground_truth.txt"
)

rgb_dir = RESULTS_DIR / "rgb_signals"
metrics_dir = RESULTS_DIR / "metrics"

bpm_path = (
    metrics_dir
    / f"subject{subject_number}_chrom_bpm.npy"
)

indices_path = (
    rgb_dir
    / f"subject{subject_number}_valid_window_indices.npy"
)


# --------------------------------------------------
# Check files
# --------------------------------------------------

if not ground_truth_path.exists():
    print("ERROR: Ground truth file not found!")
    print(ground_truth_path)
    sys.exit()

if not bpm_path.exists():
    print("ERROR: CHROM BPM file not found!")
    print(bpm_path)
    sys.exit()

if not indices_path.exists():
    print("ERROR: Valid window indices not found!")
    print(indices_path)
    sys.exit()


# --------------------------------------------------
# Load data
# --------------------------------------------------

ground_truth = np.loadtxt(
    ground_truth_path
)

estimated_bpm = np.load(
    bpm_path
)

valid_window_indices = np.load(
    indices_path
).astype(int)


print(
    f"Evaluating Subject {subject_number}"
)

print(
    "Ground truth shape:",
    ground_truth.shape
)

print(
    "Estimated BPM shape:",
    estimated_bpm.shape
)

print(
    "Valid window indices:",
    valid_window_indices
)


# --------------------------------------------------
# Extract ground-truth BPM
# --------------------------------------------------

ground_truth_bpm = ground_truth[1]


# --------------------------------------------------
# Get subject-specific FPS
# --------------------------------------------------

fps = get_subject_fps(
    subject_number
)


# --------------------------------------------------
# Calculate exact window size
# --------------------------------------------------

window_size = int(
    fps * WINDOW_SECONDS
)


print(
    f"FPS: {fps}"
)

print(
    f"Window size: {window_size} frames"
)


# --------------------------------------------------
# Calculate GT BPM for exact original windows
# --------------------------------------------------

ground_truth_windowed = []


for window_index in valid_window_indices:

    start = (
        window_index
        * window_size
    )

    end = (
        start
        + window_size
    )


    if start >= len(ground_truth_bpm):

        ground_truth_windowed.append(
            np.nan
        )

        continue


    end = min(
        end,
        len(ground_truth_bpm)
    )


    window_gt = (
        ground_truth_bpm[start:end]
    )


    # Remove invalid GT values
    window_gt = window_gt[
        np.isfinite(window_gt)
    ]


    if len(window_gt) == 0:

        ground_truth_windowed.append(
            np.nan
        )

    else:

        ground_truth_windowed.append(
            np.mean(window_gt)
        )


ground_truth_windowed = np.array(
    ground_truth_windowed
)


# --------------------------------------------------
# Check lengths
# --------------------------------------------------

if len(estimated_bpm) != len(
    ground_truth_windowed
):

    print(
        "\nERROR: Estimated BPM and "
        "ground truth lengths differ."
    )

    print(
        "Estimated BPM:",
        len(estimated_bpm)
    )

    print(
        "Ground truth:",
        len(ground_truth_windowed)
    )

    sys.exit()


# --------------------------------------------------
# Remove invalid values
# --------------------------------------------------

valid = (
    np.isfinite(estimated_bpm)
    &
    np.isfinite(ground_truth_windowed)
)


estimated_valid = (
    estimated_bpm[valid]
)

ground_truth_valid = (
    ground_truth_windowed[valid]
)


if len(estimated_valid) < 2:

    print(
        "\nERROR: Not enough valid windows "
        "for evaluation."
    )

    sys.exit()


# --------------------------------------------------
# MAE
# --------------------------------------------------

mae = np.mean(
    np.abs(
        estimated_valid
        - ground_truth_valid
    )
)


# --------------------------------------------------
# RMSE
# --------------------------------------------------

rmse = np.sqrt(
    np.mean(
        (
            estimated_valid
            - ground_truth_valid
        ) ** 2
    )
)


# --------------------------------------------------
# Pearson correlation
# --------------------------------------------------

r, p_value = pearsonr(
    estimated_valid,
    ground_truth_valid
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n" + "=" * 50)

print(
    f"Subject {subject_number} "
    "CHROM Evaluation"
)

print("=" * 50)

print(
    f"MAE  : {mae:.2f} BPM"
)

print(
    f"RMSE : {rmse:.2f} BPM"
)

print(
    f"R    : {r:.4f}"
)

print("=" * 50)


# --------------------------------------------------
# Save metrics
# --------------------------------------------------

metrics_dir.mkdir(
    parents=True,
    exist_ok=True
)

metrics_output = (
    metrics_dir
    / f"subject{subject_number}_chrom_metrics.txt"
)


with open(
    metrics_output,
    "w"
) as f:

    f.write(
        f"Subject {subject_number} "
        "CHROM Evaluation\n"
    )

    f.write(
        f"MAE  : {mae:.2f} BPM\n"
    )

    f.write(
        f"RMSE : {rmse:.2f} BPM\n"
    )

    f.write(
        f"R    : {r:.4f}\n"
    )


print(
    f"\nMetrics saved to:\n"
    f"{metrics_output}"
)