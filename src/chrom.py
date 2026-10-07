import numpy as np
import sys

from config import RESULTS_DIR


# --------------------------------------------------
# Check command-line argument
# --------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: python src/chrom.py <subject_number>")
    print("Example: python src/chrom.py 1")
    sys.exit()

subject_number = int(sys.argv[1])


# --------------------------------------------------
# Paths
# --------------------------------------------------

rgb_dir = RESULTS_DIR / "rgb_signals"
rppg_dir = RESULTS_DIR / "rppg_signals"

input_path = (
    rgb_dir
    / f"subject{subject_number}_rgb_windows.npy"
)

output_path = (
    rppg_dir
    / f"subject{subject_number}_chrom.npy"
)


# --------------------------------------------------
# Check input
# --------------------------------------------------

if not input_path.exists():
    print("ERROR: Windowed RGB file not found!")
    print(input_path)

    print(
        f"\nRun windowing first:\n"
        f"python src/windowing.py {subject_number}"
    )

    sys.exit()


# --------------------------------------------------
# Load RGB windows
# --------------------------------------------------

rgb_windows = np.load(input_path)

print(f"Processing Subject {subject_number}")
print(
    "RGB windows shape:",
    rgb_windows.shape
)


# --------------------------------------------------
# CHROM algorithm
# --------------------------------------------------

rppg_windows = []


for window in rgb_windows:

    R = window[:, 0]
    G = window[:, 1]
    B = window[:, 2]


    # ----------------------------------------------
    # CHROM projections
    # ----------------------------------------------

    X = (
        3 * R
        - 2 * G
    )

    Y = (
        1.5 * R
        + G
        - 1.5 * B
    )


    # ----------------------------------------------
    # Adaptive scaling
    # ----------------------------------------------

    std_X = np.std(X)
    std_Y = np.std(Y)

    if std_Y < 1e-12:

        alpha = 0.0

    else:

        alpha = (
            std_X
            / std_Y
        )


    # ----------------------------------------------
    # CHROM signal
    # ----------------------------------------------

    S = X - alpha * Y

    S = S - np.mean(S)


    rppg_windows.append(S)


# --------------------------------------------------
# Convert to NumPy array
# --------------------------------------------------

rppg_windows = np.array(
    rppg_windows
)


# --------------------------------------------------
# Save
# --------------------------------------------------

rppg_dir.mkdir(
    parents=True,
    exist_ok=True
)

np.save(
    output_path,
    rppg_windows
)


print("\nCHROM extraction complete!")

print(
    "rPPG windows shape:",
    rppg_windows.shape
)

print(
    f"Saved to: {output_path}"
)