import numpy as np

ground_truth_path = r"G:\.shortcut-targets-by-id\1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk\UBFC_DATASET\DATASET_2\subject1\ground_truth.txt"

# ---------------------------------
# LOAD GROUND TRUTH
# ---------------------------------

ground_truth = np.loadtxt(
    ground_truth_path
)

print("Ground truth loaded!")
print("Shape:", ground_truth.shape)

# ---------------------------------
# DISPLAY BASIC INFORMATION
# ---------------------------------

print("\nFirst 5 columns:")
print(ground_truth[:, :5])

print("\nNumber of rows:", ground_truth.shape[0])
print("Number of columns:", ground_truth.shape[1])