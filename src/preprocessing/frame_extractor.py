# src/preprocessing/frame_extractor.py

import cv2
import os
import numpy as np
from pathlib import Path

# ─── Constants ───────────────────────────────────────────────
FRAME_INTERVAL   = 10      # Extract every 10th frame (not every frame — saves time)
IMG_SIZE         = (224, 224)  # Standard input size for MobileNet / ResNet
FACE_SCALE       = 1.1     # How much to scale the detection window each step
FACE_NEIGHBORS   = 5       # How many neighbors a detection needs to be kept
MIN_FACE_SIZE    = (60, 60)    # Ignore tiny detections (likely noise)