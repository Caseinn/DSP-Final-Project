import cv2
import numpy as np

def draw_face_roi(frame, landmarks, face_regions, alpha=0.4):

    h, w, _ = frame.shape
    overlay = frame.copy()
    color = (0, 255, 0)  # Green mask in BGR

    for region_name, idxs in face_regions.items():
        pts = []
        for idx in idxs:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            pts.append((x, y))

        if len(pts) >= 3:
            pts_array = np.array(pts, dtype=np.int32)
            hull = cv2.convexHull(pts_array)
            cv2.fillConvexPoly(overlay, hull, color)

    # Apply translucent overlay
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_shoulders(frame, landmarks):
    h, w, _ = frame.shape
    for i in [11, 12]:
        if landmarks[i].visibility > 0.5:
            x = int(landmarks[i].x * w)
            y = int(landmarks[i].y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
