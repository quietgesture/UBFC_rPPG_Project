import cv2
import matplotlib.pyplot as plt

video_path = r"G:\.shortcut-targets-by-id\1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk\UBFC_DATASET\DATASET_2\subject1\vid.avi"

# Load face detector
model_path = r"models\res10_300x300_ssd_iter_140000.caffemodel"
config_path = r"models\deploy.prototxt"

net = cv2.dnn.readNet(model_path, config_path)

# Read first frame
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()

if not ret:
    print("ERROR: Could not read frame.")
    cap.release()
    exit()

height, width = frame.shape[:2]

# -----------------------------
# FACE DETECTION
# -----------------------------

blob = cv2.dnn.blobFromImage(
    frame,
    scalefactor=1.0,
    size=(300, 300),
    mean=(104.0, 177.0, 123.0)
)

net.setInput(blob)
detections = net.forward()

best_confidence = 0
best_box = None

for i in range(detections.shape[2]):

    confidence = detections[0, 0, i, 2]

    if confidence > best_confidence:

        box = detections[0, 0, i, 3:7] * [
            width,
            height,
            width,
            height
        ]

        x1, y1, x2, y2 = box.astype(int)

        best_confidence = confidence
        best_box = (x1, y1, x2, y2)

if best_box is None:
    print("No face detected.")
    cap.release()
    exit()

x1, y1, x2, y2 = best_box

print("Face detected!")
print(f"Bounding box: ({x1}, {y1}) → ({x2}, {y2})")

# -----------------------------
# SKIN ROI SELECTION
# -----------------------------

face_width = x2 - x1
face_height = y2 - y1

# Forehead
forehead = (
    int(x1 + 0.25 * face_width),
    int(y1 + 0.10 * face_height),
    int(x1 + 0.75 * face_width),
    int(y1 + 0.30 * face_height)
)

# Left cheek
left_cheek = (
    int(x1 + 0.10 * face_width),
    int(y1 + 0.45 * face_height),
    int(x1 + 0.35 * face_width),
    int(y1 + 0.70 * face_height)
)

# Right cheek
right_cheek = (
    int(x1 + 0.65 * face_width),
    int(y1 + 0.45 * face_height),
    int(x1 + 0.90 * face_width),
    int(y1 + 0.70 * face_height)
)

# Draw face
cv2.rectangle(
    frame,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    2
)

# Draw ROIs
rois = [
    ("Forehead", forehead),
    ("Left Cheek", left_cheek),
    ("Right Cheek", right_cheek)
]

for name, (rx1, ry1, rx2, ry2) in rois:

    cv2.rectangle(
        frame,
        (rx1, ry1),
        (rx2, ry2),
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        name,
        (rx1, ry1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1
    )

# Display
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 7))
plt.imshow(frame_rgb)
plt.axis("off")
plt.title("Face Detection + Skin ROIs")
plt.show()

cap.release()