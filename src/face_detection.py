import cv2
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

video_path = r"G:\.shortcut-targets-by-id\1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk\UBFC_DATASET\DATASET_2\subject1\vid.avi"

model_path = r"models\res10_300x300_ssd_iter_140000.caffemodel"
config_path = r"models\deploy.prototxt"


# --------------------------------------------------
# 2. Load the face detector
# --------------------------------------------------

net = cv2.dnn.readNetFromCaffe(
    config_path,
    model_path
)


# --------------------------------------------------
# 3. Open UBFC video
# --------------------------------------------------

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()


# --------------------------------------------------
# 4. Read first frame
# --------------------------------------------------

ret, frame = cap.read()

if not ret:
    print("ERROR: Could not read frame.")
    cap.release()
    exit()


print("Frame extracted successfully!")
print("Frame shape:", frame.shape)


# --------------------------------------------------
# 5. Prepare frame for DNN
# --------------------------------------------------

blob = cv2.dnn.blobFromImage(
    frame,
    scalefactor=1.0,
    size=(300, 300),
    mean=(104.0, 177.0, 123.0)
)


# --------------------------------------------------
# 6. Run face detector
# --------------------------------------------------

net.setInput(blob)

detections = net.forward()


# --------------------------------------------------
# 7. Find the best face
# --------------------------------------------------

height, width = frame.shape[:2]

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


# --------------------------------------------------
# 8. Draw best face
# --------------------------------------------------

if best_box is None:

    print("No face detected.")

else:

    x1, y1, x2, y2 = best_box

    print("Face detected!")
    print(f"Confidence: {best_confidence:.2f}")
    print(f"Bounding box: ({x1}, {y1}) → ({x2}, {y2})")

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )


# --------------------------------------------------
# 9. Display
# --------------------------------------------------

frame_rgb = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(10, 7))
plt.imshow(frame_rgb)
plt.axis("off")
plt.title("OpenCV DNN Face Detection")
plt.show()


# --------------------------------------------------
# 10. Close video
# --------------------------------------------------

cap.release()