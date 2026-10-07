import cv2
import numpy as np
import sys
from pathlib import Path

from config import DATASET_ROOT, RESULTS_DIR


# --------------------------------------------------
# Check command-line argument
# --------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: python src/rgb_extraction.py <subject_number>")
    print("Example: python src/rgb_extraction.py 1")
    sys.exit()

subject_number = int(sys.argv[1])


# --------------------------------------------------
# Paths
# --------------------------------------------------

subject_dir = DATASET_ROOT / f"subject{subject_number}"
video_path = subject_dir / "vid.avi"

rgb_dir = RESULTS_DIR / "rgb_signals"

output_path = (
    rgb_dir / f"subject{subject_number}_rgb.npy"
)


# --------------------------------------------------
# Check video
# --------------------------------------------------

if not video_path.exists():
    print("ERROR: Video not found!")
    print(video_path)
    sys.exit()


print(f"Processing Subject {subject_number}")
print(f"Video: {video_path}")


# --------------------------------------------------
# Open video
# --------------------------------------------------

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("ERROR: Could not open video.")
    sys.exit()


fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print(f"FPS: {fps}")
print(f"Frames: {frame_count}")


# --------------------------------------------------
# Load face detector
# --------------------------------------------------

model_path = (
    Path("models")
    / "res10_300x300_ssd_iter_140000.caffemodel"
)

config_path = (
    Path("models")
    / "deploy.prototxt"
)

net = cv2.dnn.readNetFromCaffe(
    str(config_path),
    str(model_path)
)


# --------------------------------------------------
# RGB extraction
# --------------------------------------------------

rgb_values = []

# Keep track of frames where detection fails
failed_frames = []

frame_number = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    height, width = frame.shape[:2]


    # ----------------------------------------------
    # Face detection
    # ----------------------------------------------

    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0)
    )

    net.setInput(blob)

    detections = net.forward()


    # ----------------------------------------------
    # Select highest-confidence face
    # ----------------------------------------------

    best_confidence = 0
    best_box = None

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > best_confidence:

            box = (
                detections[0, 0, i, 3:7]
                * np.array(
                    [width, height, width, height]
                )
            )

            x1, y1, x2, y2 = box.astype(int)

            best_confidence = confidence

            best_box = (
                x1,
                y1,
                x2,
                y2
            )


    # ----------------------------------------------
    # If face detection fails
    # ----------------------------------------------

    if best_box is None:

        # Store NaN so the frame position is preserved
        rgb_values.append(
            [np.nan, np.nan, np.nan]
        )

        failed_frames.append(frame_number)

        continue


    x1, y1, x2, y2 = best_box


    # ----------------------------------------------
    # Crop face
    # ----------------------------------------------

    face = frame[
        max(0, y1):min(height, y2),
        max(0, x1):min(width, x2)
    ]


    if face.size == 0:

        rgb_values.append(
            [np.nan, np.nan, np.nan]
        )

        failed_frames.append(frame_number)

        continue


    # ----------------------------------------------
    # Select facial ROI
    # ----------------------------------------------

    fh, fw = face.shape[:2]

    roi = face[
        int(0.20 * fh):int(0.70 * fh),
        int(0.20 * fw):int(0.80 * fw)
    ]


    if roi.size == 0:

        rgb_values.append(
            [np.nan, np.nan, np.nan]
        )

        failed_frames.append(frame_number)

        continue


    # ----------------------------------------------
    # Convert BGR → RGB
    # ----------------------------------------------

    roi_rgb = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2RGB
    )


    # ----------------------------------------------
    # Mean RGB
    # ----------------------------------------------

    mean_rgb = np.mean(
        roi_rgb,
        axis=(0, 1)
    )

    rgb_values.append(mean_rgb)


cap.release()


# --------------------------------------------------
# Convert to NumPy array
# --------------------------------------------------

rgb_values = np.array(
    rgb_values,
    dtype=np.float64
)


print("\nRGB extraction complete!")

print(
    "RGB shape:",
    rgb_values.shape
)

print(
    "Failed frames:",
    len(failed_frames)
)


# --------------------------------------------------
# Save
# --------------------------------------------------

rgb_dir.mkdir(
    parents=True,
    exist_ok=True
)

np.save(
    output_path,
    rgb_values
)


print(
    f"Saved to: {output_path}"
)


# --------------------------------------------------
# Warning if frames failed
# --------------------------------------------------

if len(failed_frames) > 0:

    print(
        "\nWARNING:"
        f" {len(failed_frames)} frames "
        "had no valid face/ROI."
    )

    print(
        "Their positions were preserved "
        "using NaN values."
    )

else:

    print(
        "\nAll frames had valid face detection."
    )