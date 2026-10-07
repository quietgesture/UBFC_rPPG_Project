from pathlib import Path
import json


# ==========================================
# DATASET
# ==========================================

DATASET_ROOT = Path.home() / "UBFC_GT"

FPS_MAP_PATH = Path.home() / "fps_map.json"

with open(FPS_MAP_PATH, "r") as f:
    FPS_MAP = json.load(f)


# ==========================================
# PROJECT DIRECTORIES
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

PLOTS_DIR = PROJECT_ROOT / "plots"


# ==========================================
# WINDOW SETTINGS
# ==========================================

WINDOW_SECONDS = 10


# ==========================================
# HEART RATE RANGE
# ==========================================

MIN_BPM = 42

MAX_BPM = 240

MIN_HZ = MIN_BPM / 60

MAX_HZ = MAX_BPM / 60


# ==========================================
# PSD SETTINGS
# ==========================================

NFFT = 8192


# ==========================================
# BANDPASS FILTER
# ==========================================

LOW_CUTOFF = 0.7

HIGH_CUTOFF = 4.0

FILTER_ORDER = 3


# ==========================================
# GET FPS OF A SUBJECT
# ==========================================

def get_subject_fps(subject_number):

    subject_name = f"subject{subject_number}"

    if subject_name not in FPS_MAP:
        raise ValueError(
            f"FPS not found for {subject_name}"
        )

    fps = float(FPS_MAP[subject_name])

    if fps <= 0:
        raise ValueError(
            f"Invalid FPS for {subject_name}: {fps}"
        )

    return fps
