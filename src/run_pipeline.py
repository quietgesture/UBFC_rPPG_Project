import subprocess
import sys
from pathlib import Path

from config import DATASET_ROOT


# --------------------------------------------------
# Find all actual subject folders
# --------------------------------------------------

subject_dirs = sorted(
    DATASET_ROOT.glob("subject*"),
    key=lambda p: int(
        p.name.replace("subject", "")
    )
)


if not subject_dirs:
    print("ERROR: No subject folders found!")
    sys.exit(1)


subjects = []

for subject_dir in subject_dirs:

    subject_number = int(
        subject_dir.name.replace("subject", "")
    )

    video_path = subject_dir / "vid.avi"
    gt_path = subject_dir / "ground_truth.txt"

    if video_path.exists() and gt_path.exists():

        subjects.append(subject_number)


print("=" * 60)
print("UBFC-rPPG FULL PIPELINE")
print("=" * 60)

print(
    f"Subjects found: {len(subjects)}"
)

print(
    "Subjects:",
    subjects
)

print("=" * 60)


# --------------------------------------------------
# Run one Python script
# --------------------------------------------------

def run_stage(script, subject):

    command = [
        sys.executable,
        f"src/{script}",
        str(subject)
    ]

    print("\n")
    print("=" * 60)

    print(
        f"Running {script} "
        f"for Subject {subject}"
    )

    print("=" * 60)

    result = subprocess.run(
        command
    )

    if result.returncode != 0:

        print(
            f"\nERROR: {script} failed "
            f"for Subject {subject}"
        )

        return False

    return True


# --------------------------------------------------
# Pipeline stages
# --------------------------------------------------

stages = [
    "rgb_extraction.py",
    "windowing.py",

    "pos.py",
    "filtering.py",
    "frequency.py",
    "evaluation.py",

    "chrom.py",
    "chrom_filtering.py",
    "chrom_frequency.py",
    "chrom_evaluation.py"
]


# --------------------------------------------------
# Process every subject
# --------------------------------------------------

successful_subjects = []
failed_subjects = []


for subject in subjects:

    print("\n\n")
    print("#" * 70)
    print(
        f"# STARTING SUBJECT {subject}"
    )
    print("#" * 70)


    subject_success = True


    for stage in stages:

        success = run_stage(
            stage,
            subject
        )

        if not success:

            subject_success = False

            print(
                f"\nSkipping remaining stages "
                f"for Subject {subject}"
            )

            break


    if subject_success:

        successful_subjects.append(
            subject
        )

        print(
            f"\nSubject {subject} "
            "completed successfully!"
        )

    else:

        failed_subjects.append(
            subject
        )


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n\n")

print("=" * 70)
print("FULL PIPELINE COMPLETE")
print("=" * 70)

print(
    "Successful subjects:",
    len(successful_subjects)
)

print(
    successful_subjects
)

print(
    "\nFailed subjects:",
    len(failed_subjects)
)

print(
    failed_subjects
)

print("=" * 70)