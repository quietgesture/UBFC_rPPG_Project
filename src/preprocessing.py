import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# LOAD RGB SIGNAL
# ---------------------------------

rgb_signal = np.load(
    "results/rgb_signals/subject1_rgb.npy"
)

print("Loaded RGB signal!")
print("Shape:", rgb_signal.shape)

# ---------------------------------
# NORMALIZATION
# ---------------------------------

# Remove the mean of each RGB channel
rgb_normalized = rgb_signal - np.mean(
    rgb_signal,
    axis=0
)

# Divide by standard deviation
rgb_normalized = rgb_normalized / np.std(
    rgb_normalized,
    axis=0
)

print("Preprocessing complete!")
print("Mean after normalization:")
print(np.mean(rgb_normalized, axis=0))

print("\nStandard deviation after normalization:")
print(np.std(rgb_normalized, axis=0))

# ---------------------------------
# SAVE PREPROCESSED SIGNAL
# ---------------------------------

np.save(
    "results/rgb_signals/subject1_rgb_preprocessed.npy",
    rgb_normalized
)

print("\nPreprocessed RGB signal saved!")

# ---------------------------------
# PLOT
# ---------------------------------

fps = 29.264106

time = np.arange(
    len(rgb_normalized)
) / fps

plt.figure(figsize=(12, 6))

plt.plot(
    time,
    rgb_normalized[:, 0],
    label="Red"
)

plt.plot(
    time,
    rgb_normalized[:, 1],
    label="Green"
)

plt.plot(
    time,
    rgb_normalized[:, 2],
    label="Blue"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Normalized Intensity")

plt.title("Preprocessed RGB Signals")

plt.legend()
plt.grid()

plt.show()