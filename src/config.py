from pathlib import Path

# Directory structure
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
AUGMENTED_DIR = Path("data/augmented")
REPORTS_DIR = Path("reports")

# Class definitions for defect detection
CLASSES = ["ok", "scratch", "dent", "crack", "chip", "discoloration"]
DEFECT_CLASSES = [c for c in CLASSES if c != "ok"]

# Camera and line metadata
CAMERA_IDS = ["CAM_01", "CAM_02", "CAM_03"]
LINE_IDS = ["LINE_A", "LINE_B"]
SHIFTS = ["morning", "afternoon", "night"]

# Training defaults
IMG_SIZE = 640
