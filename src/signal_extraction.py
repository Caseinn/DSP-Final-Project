# src/signal_extraction.py
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# Define face regions including both cheeks and nose
FACE_REGIONS = {
    "right_cheek": [192, 213, 215],  # Right lower cheek cluster
    "left_cheek": [416, 433, 435],   # Left lower cheek cluster
    "nose": [75, 6, 2, 305]              # Tip, base, and both side of the nose
}

def extract_rppg_roi(frame, face_result):
    """Ekstrak sinyal rPPG dari facial ROIs (pipi kanan, pipi kiri, dan hidung)."""
    if not face_result.multi_face_landmarks:
        return None, None
    
    ih, iw, _ = frame.shape
    landmarks = face_result.multi_face_landmarks[0].landmark
    regions = [
        FACE_REGIONS["right_cheek"],
        FACE_REGIONS["left_cheek"],
        FACE_REGIONS["nose"]
    ]
    
    rgbs = []
    face_points = []

    for region in regions:
        region_colors = []
        region_points = []
        for idx in region:
            x_norm, y_norm = landmarks[idx].x, landmarks[idx].y
            x, y = int(x_norm * iw), int(y_norm * ih)
            region_points.append((x, y))
            patch = frame[max(0, y-10):min(ih, y+10), max(0, x-10):min(iw, x+10)]
            if patch.size > 0:
                region_colors.append(np.mean(patch.reshape(-1, 3), axis=0))
        if region_colors:
            rgbs.append(np.mean(region_colors, axis=0))
            face_points.extend(region_points)
    
    return (np.mean(rgbs, axis=0) if rgbs else None), face_points