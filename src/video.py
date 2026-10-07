import cv2
import matplotlib.pyplot as plt

# Path to the UBFC video
video_path = r"G:\.shortcut-targets-by-id\1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk\UBFC_DATASET\DATASET_2\subject1\vid.avi"

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

# Read the first frame
ret, frame = cap.read()

if not ret:
    print("ERROR: Could not read frame.")
    cap.release()
    exit()

print("First frame extracted successfully!")
print(f"Frame shape: {frame.shape}")

# OpenCV reads images as BGR.
# Convert BGR → RGB for displaying with Matplotlib.
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Display the frame
plt.figure(figsize=(10, 7))
plt.imshow(frame_rgb)
plt.axis("off")
plt.title("UBFC Subject 1 - First Frame")
plt.show()

# Close video
cap.release()