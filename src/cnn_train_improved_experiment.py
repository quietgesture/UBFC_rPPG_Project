import numpy as np
import tensorflow as tf

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

X_PATH = Path("results/cnn/X.npy")
Y_PATH = Path("results/cnn/y.npy")
SUBJECTS_PATH = Path("results/cnn/subjects.npy")

MODEL_DIR = Path("results/cnn/model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

EPOCHS = 100
BATCH_SIZE = 16


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING IMPROVED CNN DATASET")
print("=" * 60)

X = np.load(X_PATH)
y = np.load(Y_PATH)
subjects = np.load(SUBJECTS_PATH)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("subjects shape:", subjects.shape)

print("Unique subjects:", len(np.unique(subjects)))


# ============================================================
# SUBJECT-WISE SPLIT
# ============================================================

unique_subjects = np.unique(subjects)

rng = np.random.default_rng(SEED)

shuffled_subjects = unique_subjects.copy()
rng.shuffle(shuffled_subjects)

n_subjects = len(shuffled_subjects)

n_train = int(0.70 * n_subjects)
n_val = int(0.15 * n_subjects)

train_subjects = shuffled_subjects[:n_train]

val_subjects = shuffled_subjects[
    n_train:n_train + n_val
]

test_subjects = shuffled_subjects[
    n_train + n_val:
]


print("\n" + "=" * 60)
print("SUBJECT SPLIT")
print("=" * 60)

print("Train subjects:", len(train_subjects))
print("Validation subjects:", len(val_subjects))
print("Test subjects:", len(test_subjects))

print("Train IDs:", train_subjects)
print("Validation IDs:", val_subjects)
print("Test IDs:", test_subjects)


# ============================================================
# CREATE MASKS
# ============================================================

train_mask = np.isin(
    subjects,
    train_subjects
)

val_mask = np.isin(
    subjects,
    val_subjects
)

test_mask = np.isin(
    subjects,
    test_subjects
)


X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

X_test = X[test_mask]
y_test = y[test_mask]

subjects_test = subjects[test_mask]


print("\n" + "=" * 60)
print("DATA SPLIT")
print("=" * 60)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


print("\nHR distribution:")

print(
    f"Train HR: mean={y_train.mean():.2f}, "
    f"std={y_train.std():.2f}, "
    f"min={y_train.min():.2f}, "
    f"max={y_train.max():.2f}"
)

print(
    f"Val HR:   mean={y_val.mean():.2f}, "
    f"std={y_val.std():.2f}, "
    f"min={y_val.min():.2f}, "
    f"max={y_val.max():.2f}"
)

print(
    f"Test HR:  mean={y_test.mean():.2f}, "
    f"std={y_test.std():.2f}, "
    f"min={y_test.min():.2f}, "
    f"max={y_test.max():.2f}"
)


# ============================================================
# INPUT NORMALIZATION
# ============================================================

# Calculate statistics ONLY from training data.

input_mean = X_train.mean(
    axis=(0, 1),
    keepdims=True
)

input_std = X_train.std(
    axis=(0, 1),
    keepdims=True
)

input_std = np.maximum(
    input_std,
    1e-8
)


X_train = (
    X_train - input_mean
) / input_std

X_val = (
    X_val - input_mean
) / input_std

X_test = (
    X_test - input_mean
) / input_std


print("\n" + "=" * 60)
print("INPUT NORMALIZATION")
print("=" * 60)

print("Input mean:", input_mean.flatten())
print("Input std :", input_std.flatten())

print(
    "X_train normalized mean:",
    X_train.mean()
)

print(
    "X_train normalized std:",
    X_train.std()
)


# ============================================================
# TARGET NORMALIZATION
# ============================================================

# Normalize HR using ONLY training targets.

target_mean = y_train.mean()
target_std = y_train.std()

target_std = max(
    target_std,
    1e-8
)


y_train_norm = (
    y_train - target_mean
) / target_std

y_val_norm = (
    y_val - target_mean
) / target_std

y_test_norm = (
    y_test - target_mean
) / target_std


print("\n" + "=" * 60)
print("TARGET NORMALIZATION")
print("=" * 60)

print(
    f"Target mean: {target_mean:.6f}"
)

print(
    f"Target std : {target_std:.6f}"
)

print(
    "Normalized train target mean:",
    y_train_norm.mean()
)

print(
    "Normalized train target std:",
    y_train_norm.std()
)


# ============================================================
# CNN MODEL
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(300, 3)
    ),

    tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=5,
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling1D(
        pool_size=2
    ),

    tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=5,
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling1D(
        pool_size=2
    ),

    tf.keras.layers.Conv1D(
        filters=128,
        kernel_size=5,
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.3
    ),

    tf.keras.layers.Dense(
        1
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="mse",

    metrics=[
        tf.keras.metrics.MeanAbsoluteError(
            name="mae"
        )
    ]
)


print("\n" + "=" * 60)
print("CNN MODEL")
print("=" * 60)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint_path = (
    MODEL_DIR /
    "best_model.keras"
)

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        save_best_only=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = model.fit(

    X_train,
    y_train_norm,

    validation_data=(
        X_val,
        y_val_norm
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("TEST EVALUATION")
print("=" * 60)

predictions_norm = model.predict(
    X_test,
    verbose=0
).flatten()


# ============================================================
# CONVERT PREDICTIONS BACK TO BPM
# ============================================================

predictions = (
    predictions_norm *
    target_std
) + target_mean


# ============================================================
# METRICS
# ============================================================

errors = predictions - y_test

mae = np.mean(
    np.abs(errors)
)

rmse = np.sqrt(
    np.mean(
        errors ** 2
    )
)

if len(y_test) > 1:

    correlation = np.corrcoef(
        y_test,
        predictions
    )[0, 1]

else:

    correlation = np.nan


print(
    f"Test MAE  : {mae:.4f} BPM"
)

print(
    f"Test RMSE : {rmse:.4f} BPM"
)

print(
    f"Test R    : {correlation:.4f}"
)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION DISTRIBUTION")
print("=" * 60)

print(
    f"GT mean/std: "
    f"{y_test.mean():.4f} / "
    f"{y_test.std():.4f}"
)

print(
    f"Pred mean/std: "
    f"{predictions.mean():.4f} / "
    f"{predictions.std():.4f}"
)

print(
    f"GT range: "
    f"{y_test.min():.4f} - "
    f"{y_test.max():.4f}"
)

print(
    f"Pred range: "
    f"{predictions.min():.4f} - "
    f"{predictions.max():.4f}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

np.save(
    MODEL_DIR / "test_predictions.npy",
    predictions
)

np.save(
    MODEL_DIR / "test_ground_truth.npy",
    y_test
)

np.save(
    MODEL_DIR / "test_subjects.npy",
    subjects_test
)


# ============================================================
# SAVE INPUT NORMALIZATION PARAMETERS
# ============================================================

np.save(
    MODEL_DIR / "input_mean.npy",
    input_mean
)

np.save(
    MODEL_DIR / "input_std.npy",
    input_std
)


# ============================================================
# SAVE TARGET NORMALIZATION PARAMETERS
# ============================================================

np.save(
    MODEL_DIR / "target_mean.npy",
    np.array(target_mean)
)

np.save(
    MODEL_DIR / "target_std.npy",
    np.array(target_std)
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    MODEL_DIR / "final_model.keras"
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

np.save(
    MODEL_DIR / "train_loss.npy",
    np.asarray(history.history["loss"])
)

np.save(
    MODEL_DIR / "val_loss.npy",
    np.asarray(history.history["val_loss"])
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    "Best model:",
    checkpoint_path
)

print(
    "Final model:",
    MODEL_DIR / "final_model.keras"
)

print(
    "Predictions:",
    MODEL_DIR / "test_predictions.npy"
)

print(
    "Ground truth:",
    MODEL_DIR / "test_ground_truth.npy"
)

print(
    "Test subjects:",
    MODEL_DIR / "test_subjects.npy"
)
