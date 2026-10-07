import numpy as np
import matplotlib.pyplot as plt
import sys

from config import (
    RESULTS_DIR,
    WINDOW_SECONDS,
    get_subject_fps
)


# --------------------------------------------------
# Check command-line argument
# --------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: python src/windowing.py <subject_number>")
    print("Example: python src/windowing.py 1")
    sys.exit()

subject_number = int(sys.argv[1])


# --------------------------------------------------
# Get subject-specific FPS
# --------------------------------------------------

fps = get_subject_fps(subject_number)

print(f"Processing Subject {subject_number}")
print(f"FPS: {fps}")
print(f"Window duration: {WINDOW_SECONDS} seconds")


# --------------------------------------------------
# Paths
# --------------------------------------------------

rgb_dir = RESULTS_DIR / "rgb_signals"

input_path = (
    rgb_dir
    / f"subject{subject_number}_rgb.npy"
)

output_path = (
    rgb_dir
    / f"subject{subject_number}_rgb_windows.npy"
)

indices_output_path = (
    rgb_dir
    / f"subject{subject_number}_valid_window_indices.npy"
)


# --------------------------------------------------
# Check input
# --------------------------------------------------

if not input_path.exists():
    print("ERROR: RGB signal file not found!")
    print(input_path)

    print(
        f"\nRun RGB extraction first:\n"
        f"python src/rgb_extraction.py {subject_number}"
    )

    sys.exit()


# --------------------------------------------------
# Load RGB signal
# --------------------------------------------------

rgb_signal = np.load(input_path)

print("RGB shape:", rgb_signal.shape)


# --------------------------------------------------
# Calculate window size
# --------------------------------------------------

window_size = int(
    fps * WINDOW_SECONDS
)

print(
    f"Window size: {window_size} frames"
)


# --------------------------------------------------
# Number of complete windows
# --------------------------------------------------

num_complete_windows = (
    len(rgb_signal) // window_size
)

print(
    f"Complete windows available: "
    f"{num_complete_windows}"
)


# --------------------------------------------------
# Create windows
# --------------------------------------------------

windows = []

window_indices = []

rejected_windows = []


for i in range(num_complete_windows):

    start = i * window_size
    end = start + window_size

    window = rgb_signal[
        start:end
    ].copy()


    # ----------------------------------------------
    # Check for missing frames
    # ----------------------------------------------

    nan_mask = np.isnan(window).any(
        axis=1
    )

    nan_count = np.sum(
        nan_mask
    )

    missing_ratio = (
        nan_count / window_size
    )


    if nan_count > 0:

        print(
            f"Window {i + 1}: "
            f"{nan_count} missing frames "
            f"({missing_ratio * 100:.2f}%)"
        )


    # ----------------------------------------------
    # Reject if >10% frames are missing
    # ----------------------------------------------

    if missing_ratio > 0.10:

        print(
            f"Window {i + 1}: REJECTED"
        )

        rejected_windows.append(i)

        continue


    # ----------------------------------------------
    # Interpolate missing RGB values
    # ----------------------------------------------

    interpolation_failed = False


    for channel in range(3):

        channel_data = (
            window[:, channel]
        )

        valid = np.isfinite(
            channel_data
        )


        if np.sum(valid) < 2:

            interpolation_failed = True

            break


        missing = ~valid


        if np.any(missing):

            channel_data[missing] = np.interp(
                np.flatnonzero(missing),
                np.flatnonzero(valid),
                channel_data[valid]
            )

        window[:, channel] = (
            channel_data
        )


    # ----------------------------------------------
    # Reject if interpolation failed
    # ----------------------------------------------

    if interpolation_failed:

        print(
            f"Window {i + 1}: "
            "REJECTED - interpolation failed"
        )

        rejected_windows.append(i)

        continue


    # ----------------------------------------------
    # Window-level normalization
    # ----------------------------------------------

    mean_rgb = np.mean(
        window,
        axis=0
    )


    # Safety check
    if np.any(mean_rgb <= 0):

        print(
            f"Window {i + 1}: "
            "REJECTED - invalid RGB mean"
        )

        rejected_windows.append(i)

        continue


    normalized_window = window


    # ----------------------------------------------
    # Save valid window
    # ----------------------------------------------

    windows.append(
        normalized_window
    )

    # IMPORTANT:
    # Save the ORIGINAL window index.
    window_indices.append(i)


# --------------------------------------------------
# Convert to NumPy arrays
# --------------------------------------------------

windows = np.array(
    windows,
    dtype=np.float64
)

window_indices = np.array(
    window_indices,
    dtype=int
)


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

rgb_dir.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Save windows
# --------------------------------------------------

np.save(
    output_path,
    windows
)


# --------------------------------------------------
# Save original window indices
# --------------------------------------------------

np.save(
    indices_output_path,
    window_indices
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nWindowing complete!")

print(
    f"Total windows: "
    f"{num_complete_windows}"
)

print(
    f"Valid windows: "
    f"{len(windows)}"
)

print(
    f"Rejected windows: "
    f"{len(rejected_windows)}"
)

print(
    f"Windows shape: "
    f"{windows.shape}"
)

print(
    f"Window indices: "
    f"{window_indices}"
)

print(
    f"\nSaved windows to:\n"
    f"{output_path}"
)

print(
    f"\nSaved window indices to:\n"
    f"{indices_output_path}"
)


# --------------------------------------------------
# Plot first valid window
# --------------------------------------------------

if len(windows) > 0:

    first_window = windows[0]

    time = (
        np.arange(window_size)
        / fps
    )


    plt.figure(
        figsize=(10, 5)
    )


    plt.plot(
        time,
        first_window[:, 0],
        label="R"
    )

    plt.plot(
        time,
        first_window[:, 1],
        label="G"
    )

    plt.plot(
        time,
        first_window[:, 2],
        label="B"
    )


    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Normalized RGB"
    )

    plt.title(
        f"Subject {subject_number} - "
        "First Normalized RGB Window"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
