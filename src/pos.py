import numpy as np
import sys

from config import RESULTS_DIR


# --------------------------------------------------
# Check command-line argument
# --------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: python src/pos.py <subject_number>")
    print("Example: python src/pos.py 1")
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
    / f"subject{subject_number}_pos.npy"
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
print("RGB windows shape:", rgb_windows.shape)


# --------------------------------------------------
# POS algorithm
# --------------------------------------------------

rppg_windows = []


for window in rgb_windows:

    R = window[:, 0]
    G = window[:, 1]
    B = window[:, 2]


    # ----------------------------------------------
    # POS projection
    # ----------------------------------------------

    S1 = G - B

    S2 = G + B - 2 * R


    # ----------------------------------------------
    # Adaptive scaling
    # ----------------------------------------------

    std_S1 = np.std(S1)
    std_S2 = np.std(S2)

    if std_S2 < 1e-12:
        alpha = 0.0
    else:
        alpha = std_S1 / std_S2


    # ----------------------------------------------
    # POS signal
    # ----------------------------------------------

    S = S1 + alpha * S2

    S = S - np.mean(S)


    rppg_windows.append(S)


# --------------------------------------------------
# Convert to NumPy array
# --------------------------------------------------

rppg_windows = np.array(rppg_windows)


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


print("\nPOS extraction complete!")

print(
    "rPPG windows shape:",
    rppg_windows.shape
)

print(
    f"Saved to: {output_path}"
)