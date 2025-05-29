# src/utils.py

import numpy as np
from datetime import datetime
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2  

FACE_REGIONS = {
    "right_cheek": [36, 50, 101, 118, 117, 116, 123, 147, 187, 205],
    "left_cheek": [266, 330, 347, 346, 345, 352, 376, 411, 425, 280],
    "forehead": [10, 109, 338, 108, 107 , 9, 336, 337, 151]
}

def extract_face_roi_rgb(frame, landmarks, region_ids):
    """
    Mengekstrak nilai rata-rata RGB dari wilayah tertentu di wajah berdasarkan landmark MediaPipe.
    
    Args:
        frame (np.ndarray): Gambar frame dari kamera
        landmarks (list): Daftar landmark wajah dari MediaPipe
        region_ids (list): ID landmark yang merepresentasikan area tertentu
    
    Returns:
        np.ndarray: Nilai rata-rata R, G, B dari area tersebut
    """
    h, w, _ = frame.shape
    pixels = []
    for idx in region_ids:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        pixels.append(frame[y, x])
    return np.mean(pixels, axis=0)  # R, G, B average

def extract_shoulder_distance(landmarks, min_visibility=0.3):
    """
    Menghitung jarak 3D antara kedua bahu (X, Y, Z) jika visibilitas cukup.
    
    Args:
        landmarks (list): Landmark pose dari MediaPipe
        min_visibility (float): Ambang minimum visibilitas bahu

    Returns:
        float or None: Jarak 3D antar bahu, atau None jika visibilitas terlalu rendah
    """
    left = landmarks[11]
    right = landmarks[12]
    if left.visibility > min_visibility and right.visibility > min_visibility:
        left_xyz = np.array([left.x, left.y, left.z])
        right_xyz = np.array([right.x, right.y, right.z])
        return np.linalg.norm(left_xyz - right_xyz)
    return None


def draw_face_roi(frame, landmarks, face_regions, alpha=0.4):
    """
    Menggambar area ROI wajah pada frame video sebagai overlay transparan.
    
    Args:
        frame (np.ndarray): Frame video yang sedang diproses
        landmarks (list): Landmark wajah dari MediaPipe
        face_regions (dict): Wilayah ROI wajah dengan daftar ID landmark
        alpha (float): Transparansi overlay (0-1)
        
    Returns:
        None: Fungsi ini mengubah frame secara langsung
    """
    h, w, _ = frame.shape
    overlay = frame.copy()
    fill_color = (0, 255, 0)
    outline_color = (0, 100, 0)

    for region_name, idxs in face_regions.items():
        pts = []
        for idx in idxs:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            pts.append((x, y))
        if len(pts) >= 3:
            pts_array = np.array(pts, dtype=np.int32)
            hull = cv2.convexHull(pts_array)
            cv2.fillConvexPoly(overlay, hull, fill_color)
            cv2.polylines(overlay, [hull], isClosed=True, color=outline_color, thickness=2)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_shoulders(frame, landmarks):
    """
    Menggambar titik bahu pada frame video jika terdeteksi.
    
    Args:
        frame (np.ndarray): Frame video
        landmarks (list): Landmark tubuh dari MediaPipe
        
    Returns:
        None: Fungsi ini mengubah frame secara langsung
    """
    h, w, _ = frame.shape
    for i in [11, 12]:
        if landmarks[i].visibility > 0.5:
            x = int(landmarks[i].x * w)
            y = int(landmarks[i].y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)