import numpy as np
from scipy.signal import periodogram
import sys

from config import (
    RESULTS_DIR,
    MIN_HZ,
    MAX_HZ,
    NFFT,
    get_subject_fps
)


# --------------------------------------------------
# Check command-line argument
# --------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: python src/frequency.py <subject_number>")
    print("Example: python src/frequency.py 1")
    sys.exit()


subject_number = int(sys.argv[1])


# --------------------------------------------------
# Get subject-specific FPS
# --------------------------------------------------

fps = get_subject_fps(subject_number)

print(f"Processing Subject {subject_number}")
print(f"FPS: {fps}")
print(
    f"Heart-rate range: "
    f"{MIN_HZ * 60:.0f} - {MAX_HZ * 60:.0f} BPM"
)
print(f"NFFT: {NFFT}")


# --------------------------------------------------
# Paths
# --------------------------------------------------

rppg_dir = RESULTS_DIR / "rppg_signals"
metrics_dir = RESULTS_DIR / "metrics"

input_path = (
    rppg_dir
    / f"subject{subject_number}_pos_filtered.npy"
)

output_path = (
    metrics_dir
    / f"subject{subject_number}_pos_bpm.npy"
)


# --------------------------------------------------
# Check input
# --------------------------------------------------

if not input_path.exists():
    print("ERROR: Filtered POS signal not found!")
    print(input_path)

    print(
        f"\nRun filtering first:\n"
        f"python src/filtering.py {subject_number}"
    )

    sys.exit()


# --------------------------------------------------
# Load filtered POS signal
# --------------------------------------------------

filtered_windows = np.load(input_path)

print(
    "Filtered signal shape:",
    filtered_windows.shape
)


# --------------------------------------------------
# Parabolic peak interpolation
# --------------------------------------------------

def interpolate_peak(
    frequencies,
    power,
    peak_index
):

    if (
        peak_index <= 0
        or peak_index >= len(power) - 1
    ):
        return frequencies[peak_index]

    y1 = power[peak_index - 1]
    y2 = power[peak_index]
    y3 = power[peak_index + 1]

    denominator = (
        y1
        - 2 * y2
        + y3
    )

    if abs(denominator) < 1e-12:
        return frequencies[peak_index]

    delta = (
        0.5
        * (y1 - y3)
        / denominator
    )

    frequency_step = (
        frequencies[1]
        - frequencies[0]
    )

    return (
        frequencies[peak_index]
        + delta * frequency_step
    )


# --------------------------------------------------
# Estimate BPM for each window
# --------------------------------------------------

estimated_bpm = []


for window_number, signal in enumerate(
    filtered_windows,
    start=1
):

    # ----------------------------------------------
    # Periodogram
    # ----------------------------------------------

    frequencies, power = periodogram(
        signal,
        fs=fps,
        nfft=NFFT
    )


    # ----------------------------------------------
    # Restrict to heart-rate range
    # ----------------------------------------------

    valid = (
        (frequencies >= MIN_HZ)
        & (frequencies <= MAX_HZ)
    )

    valid_frequencies = frequencies[valid]
    valid_power = power[valid]


    if len(valid_power) == 0:
        print(
            f"Window {window_number}: "
            "No valid frequency range"
        )

        estimated_bpm.append(np.nan)
        continue


    # ----------------------------------------------
    # Find dominant frequency
    # ----------------------------------------------

    peak_index = np.argmax(valid_power)


    # ----------------------------------------------
    # Refine peak using interpolation
    # ----------------------------------------------

    peak_frequency = interpolate_peak(
        valid_frequencies,
        valid_power,
        peak_index
    )


    # ----------------------------------------------
    # Convert Hz → BPM
    # ----------------------------------------------

    bpm = peak_frequency * 60

    estimated_bpm.append(bpm)

    print(
        f"Window {window_number}: "
        f"{bpm:.2f} BPM"
    )


# --------------------------------------------------
# Convert to NumPy array
# --------------------------------------------------

estimated_bpm = np.array(
    estimated_bpm
)


# --------------------------------------------------
# Save results
# --------------------------------------------------

metrics_dir.mkdir(
    parents=True,
    exist_ok=True
)

np.save(
    output_path,
    estimated_bpm
)


print("\nFrequency estimation complete!")

print(
    "BPM shape:",
    estimated_bpm.shape
)

print(
    f"Saved to: {output_path}"
)