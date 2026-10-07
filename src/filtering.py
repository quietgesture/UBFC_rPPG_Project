import numpy as np
from scipy.signal import butter, filtfilt
import sys

from config import (
    RESULTS_DIR,
    LOW_CUTOFF,
    HIGH_CUTOFF,
    FILTER_ORDER,
    get_subject_fps
)


# --------------------------------------------------
# Check command-line argument
# --------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: python src/filtering.py <subject_number>")
    print("Example: python src/filtering.py 1")
    sys.exit()


subject_number = int(sys.argv[1])


# --------------------------------------------------
# Get subject-specific FPS
# --------------------------------------------------

fps = get_subject_fps(subject_number)

print(f"Processing Subject {subject_number}")
print(f"FPS: {fps}")
print(
    f"Bandpass: {LOW_CUTOFF} - {HIGH_CUTOFF} Hz"
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

rppg_dir = RESULTS_DIR / "rppg_signals"

input_path = (
    rppg_dir
    / f"subject{subject_number}_pos.npy"
)

output_path = (
    rppg_dir
    / f"subject{subject_number}_pos_filtered.npy"
)


# --------------------------------------------------
# Check input
# --------------------------------------------------

if not input_path.exists():
    print("ERROR: POS signal not found!")
    print(input_path)

    print(
        f"\nRun POS first:\n"
        f"python src/pos.py {subject_number}"
    )

    sys.exit()


# --------------------------------------------------
# Load POS signal
# --------------------------------------------------

rppg_windows = np.load(input_path)

print(
    "POS signal shape:",
    rppg_windows.shape
)


# --------------------------------------------------
# Design Butterworth bandpass filter
# --------------------------------------------------

nyquist = 0.5 * fps

low = LOW_CUTOFF / nyquist
high = HIGH_CUTOFF / nyquist

b, a = butter(
    FILTER_ORDER,
    [low, high],
    btype="band"
)


# --------------------------------------------------
# Apply filter to every window
# --------------------------------------------------

filtered_windows = []

for signal in rppg_windows:

    filtered_signal = filtfilt(
        b,
        a,
        signal
    )

    filtered_windows.append(
        filtered_signal
    )


filtered_windows = np.array(
    filtered_windows
)


# --------------------------------------------------
# Save
# --------------------------------------------------

np.save(
    output_path,
    filtered_windows
)


print("\nFiltering complete!")

print(
    "Filtered signal shape:",
    filtered_windows.shape
)

print(
    f"Saved to: {output_path}"
)