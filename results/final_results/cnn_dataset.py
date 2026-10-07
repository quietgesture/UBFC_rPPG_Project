import numpy as np
from pathlib import Path

from config import DATASET_ROOT, get_subject_fps


# ============================================================
# CONFIGURATION
# ============================================================

RGB_DIR = Path("results/rgb_signals")
OUTPUT_DIR = Path("results/cnn")

TARGET_LENGTH = 300

WINDOW_SECONDS = 10

MIN_VALID_HR = 40
MAX_VALID_HR = 240


# ============================================================
# RESAMPLE ONE RGB WINDOW
# ============================================================

def resample_window(window, target_length=TARGET_LENGTH):

    old_length = window.shape[0]

    old_x = np.linspace(
        0,
        1,
        old_length
    )

    new_x = np.linspace(
        0,
        1,
        target_length
    )

    resampled = np.zeros(
        (target_length, 3),
        dtype=np.float32
    )

    for channel in range(3):

        resampled[:, channel] = np.interp(
            new_x,
            old_x,
            window[:, channel]
        )

    return resampled


# ============================================================
# NORMALIZE ONE RGB WINDOW
# ============================================================

def normalize_window(window):

    window = window.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Calculate mean RGB value for this window
    # --------------------------------------------------------

    channel_mean = np.mean(
        window,
        axis=0,
        keepdims=True
    )

    # Prevent division by zero
    channel_mean = np.maximum(
        channel_mean,
        1e-8
    )

    # --------------------------------------------------------
    # Relative RGB variation
    #
    # This removes absolute illumination/color scale while
    # preserving temporal changes in each RGB channel.
    # --------------------------------------------------------

    normalized = (
        window - channel_mean
    ) / channel_mean

    return normalized.astype(
        np.float32
    )


# ============================================================
# LOAD GROUND TRUTH HR
# ============================================================

def load_hr(subject_number):

    subject_dir = (
        DATASET_ROOT /
        f"subject{subject_number}"
    )

    gt_path = subject_dir / "ground_truth.txt"

    if not gt_path.exists():

        raise FileNotFoundError(
            f"Ground truth not found: {gt_path}"
        )

    gt = np.loadtxt(
        gt_path
    )

    # Row 1 = heart rate
    hr = gt[1]

    return hr


# ============================================================
# PROCESS ONE SUBJECT
# ============================================================

def process_subject(subject_number):

    rgb_path = (
        RGB_DIR /
        f"subject{subject_number}_rgb_windows.npy"
    )

    if not rgb_path.exists():

        print(
            f"Skipping Subject {subject_number}: "
            "RGB windows not found"
        )

        return [], []


    rgb_windows = np.load(
        rgb_path
    )


    # --------------------------------------------------------
    # Load valid window indices
    # --------------------------------------------------------

    indices_path = (
        RGB_DIR /
        f"subject{subject_number}_valid_window_indices.npy"
    )

    if not indices_path.exists():

        print(
            f"Skipping Subject {subject_number}: "
            "valid window indices not found"
        )

        return [], []


    window_indices = np.load(
        indices_path
    )


    hr = load_hr(
        subject_number
    )

    fps = get_subject_fps(
        subject_number
    )


    print(
        f"\nSubject {subject_number}"
    )

    print(
        f"RGB windows: {rgb_windows.shape}"
    )

    print(
        f"Valid window indices: {window_indices}"
    )

    print(
        f"GT HR length: {len(hr)}"
    )

    print(
        f"FPS: {fps:.6f}"
    )


    X_subject = []
    y_subject = []


    # --------------------------------------------------------
    # Process every RGB window
    # --------------------------------------------------------

    for array_index, window in enumerate(
        rgb_windows
    ):

        # ----------------------------------------------------
        # Original window index before invalid windows
        # were removed
        # ----------------------------------------------------

        if array_index >= len(
            window_indices
        ):

            print(
                f"  Stopping Subject {subject_number}: "
                "window index information is incomplete"
            )

            break


        window_index = int(
            window_indices[array_index]
        )


        original_length = window.shape[0]


        # ----------------------------------------------------
        # Determine corresponding GT frames
        #
        # RGB extraction uses approximately 10-second windows.
        # We therefore use the same original RGB window length
        # for the corresponding GT segment.
        # ----------------------------------------------------

        window_size = int(
            round(
                fps *
                WINDOW_SECONDS
            )
        )


        start_frame = (
            window_index *
            window_size
        )


        end_frame = min(
            start_frame + window_size,
            len(hr)
        )


        if start_frame >= len(hr):

            print(
                f"  Stopping Subject {subject_number}: "
                f"GT starts beyond available length "
                f"(start={start_frame}, GT={len(hr)})"
            )

            break


        hr_segment = hr[
            start_frame:end_frame
        ]


        if len(hr_segment) == 0:

            continue


        # ----------------------------------------------------
        # Validate ground-truth HR
        # ----------------------------------------------------

        valid_hr = np.all(
            (hr_segment >= MIN_VALID_HR) &
            (hr_segment <= MAX_VALID_HR)
        )


        if not valid_hr:

            print(
                f"  Rejecting window {window_index}: "
                f"invalid GT HR "
                f"(min={hr_segment.min():.2f}, "
                f"max={hr_segment.max():.2f})"
            )

            continue


        # ----------------------------------------------------
        # Average valid HR over this window
        # ----------------------------------------------------

        bpm = float(
            np.mean(
                hr_segment
            )
        )


        # ----------------------------------------------------
        # Resample RGB to common length
        # ----------------------------------------------------

        window_resampled = resample_window(
            window
        )


        # ----------------------------------------------------
        # Normalize RGB signal
        # ----------------------------------------------------

        window_normalized = normalize_window(
            window_resampled
        )


        # ----------------------------------------------------
        # Check for invalid numerical values
        # ----------------------------------------------------

        if not np.all(
            np.isfinite(
                window_normalized
            )
        ):

            print(
                f"  Rejecting window {window_index}: "
                "non-finite RGB values"
            )

            continue


        X_subject.append(
            window_normalized
        )

        y_subject.append(
            bpm
        )


    return X_subject, y_subject


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CNN DATASET CREATION - IMPROVED RGB")
    print("=" * 60)

    print(
        "Window duration:",
        WINDOW_SECONDS,
        "seconds"
    )

    print(
        "Target length:",
        TARGET_LENGTH
    )

    print(
        "Input normalization:",
        "per-window relative RGB"
    )


    all_X = []
    all_y = []
    all_subjects = []


    # --------------------------------------------------------
    # Find RGB subject files
    # --------------------------------------------------------

    rgb_files = sorted(
        RGB_DIR.glob(
            "*_rgb_windows.npy"
        ),
        key=lambda p: int(
            p.stem
            .replace(
                "subject",
                ""
            )
            .replace(
                "_rgb_windows",
                ""
            )
        )
    )


    print(
        f"RGB subject files found: "
        f"{len(rgb_files)}"
    )


    # --------------------------------------------------------
    # Process every subject
    # --------------------------------------------------------

    for rgb_file in rgb_files:

        subject_number = int(
            rgb_file.stem
            .replace(
                "subject",
                ""
            )
            .replace(
                "_rgb_windows",
                ""
            )
        )


        X_subject, y_subject = process_subject(
            subject_number
        )


        all_X.extend(
            X_subject
        )

        all_y.extend(
            y_subject
        )

        all_subjects.extend(
            [subject_number] *
            len(X_subject)
        )


    # --------------------------------------------------------
    # Convert to NumPy arrays
    # --------------------------------------------------------

    X = np.asarray(
        all_X,
        dtype=np.float32
    )

    y = np.asarray(
        all_y,
        dtype=np.float32
    )

    subjects = np.asarray(
        all_subjects,
        dtype=np.int32
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    np.save(
        OUTPUT_DIR / "X.npy",
        X
    )

    np.save(
        OUTPUT_DIR / "y.npy",
        y
    )

    np.save(
        OUTPUT_DIR / "subjects.npy",
        subjects
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CNN DATASET CREATED")
    print("=" * 60)


    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    print(
        "subjects shape:",
        subjects.shape
    )


    if len(y) > 0:

        print(
            "BPM range:",
            y.min(),
            "to",
            y.max()
        )

        print(
            "BPM mean:",
            y.mean()
        )

        print(
            "BPM std:",
            y.std()
        )

        print(
            "Unique subjects:",
            len(
                np.unique(
                    subjects
                )
            )
        )


        # ----------------------------------------------------
        # Dataset statistics by subject
        # ----------------------------------------------------

        print(
            "\nSubject-wise window counts:"
        )

        for subject in np.unique(
            subjects
        ):

            mask = (
                subjects == subject
            )

            print(
                f"Subject {int(subject):2d}: "
                f"N={int(mask.sum()):2d} "
                f"mean={y[mask].mean():.2f} "
                f"std={y[mask].std():.2f}"
            )


        # ----------------------------------------------------
        # Verify dataset values
        # ----------------------------------------------------

        print(
            "\nInput statistics:"
        )

        print(
            "X min:",
            X.min()
        )

        print(
            "X max:",
            X.max()
        )

        print(
            "X mean:",
            X.mean()
        )

        print(
            "X std:",
            X.std()
        )

        print(
            "Finite X:",
            np.all(
                np.isfinite(X)
            )
        )

        print(
            "Finite y:",
            np.all(
                np.isfinite(y)
            )
        )


    print(
        "\nSaved to:"
    )

    print(
        "results/cnn/X.npy"
    )

    print(
        "results/cnn/y.npy"
    )

    print(
        "results/cnn/subjects.npy"
    )


if __name__ == "__main__":

    main()
