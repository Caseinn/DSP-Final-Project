# File: src/main.py

import cv2
import numpy as np
import mediapipe as mp
import time
from countdown import start_countdown
from utils import FACE_REGIONS, extract_face_roi_rgb, extract_shoulder_distance, plot_signal, interpolate_nans
from visualizer import draw_face_roi, draw_shoulders
from signal_processing import apply_bandpass_filter, cpu_POS, estimate_bpm, estimate_brpm

# Init Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# Parameters
FPS = 30
DURATION = 10  # seconds
FRAME_COUNT = FPS * DURATION

# Data Storage
rppg_rgb_frames = []
resp_raw = []

# Countdown
start_countdown(3)

# Open Webcam
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter('output/video_capture.avi', fourcc, FPS, (width, height))

try:
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh, \
         mp_pose.Pose(static_image_mode=False) as pose:

        print("[INFO] Start capturing for {} seconds...".format(DURATION))

        while len(rppg_rgb_frames) < FRAME_COUNT:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_result = face_mesh.process(rgb)
            pose_result = pose.process(rgb)

            # ROI extraction
            if face_result.multi_face_landmarks:
                landmarks = face_result.multi_face_landmarks[0].landmark
                region_rgb = []
                for region in FACE_REGIONS.values():
                    roi_rgb = extract_face_roi_rgb(frame, landmarks, region)
                    region_rgb.append(roi_rgb)
                avg_rgb = np.mean(region_rgb, axis=0)
                rppg_rgb_frames.append(avg_rgb)
                draw_face_roi(frame, landmarks, FACE_REGIONS)
            else:
                rppg_rgb_frames.append(np.array([np.nan, np.nan, np.nan]))

            # Respiratory signal
            if pose_result.pose_landmarks:
                landmarks_pose = pose_result.pose_landmarks.landmark
                distance = extract_shoulder_distance(landmarks_pose)
                if distance:
                    resp_raw.append(distance)
                else:
                    resp_raw.append(np.nan)
                draw_shoulders(frame, landmarks_pose)
            else:
                resp_raw.append(np.nan)

            frame_number = len(rppg_rgb_frames)
            progress_text = f"{frame_number}/{FRAME_COUNT}"
            cv2.putText(frame, 
                        progress_text,
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 255, 255), 1)

            # Display and record
            cv2.imshow("DSP Final Project", frame)
            out_video.write(frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Capture manually stopped.")
                break

finally:
    print("[INFO] Capture finished.")
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

# Convert lists to numpy arrays
rppg_rgb_frames = np.array(rppg_rgb_frames)  # Shape: [frames, 3]
resp_raw = np.array(resp_raw)               # Shape: [frames]

# Check if no valid data collected
if len(rppg_rgb_frames) == 0 or np.all(np.isnan(rppg_rgb_frames)):
    print("[ERROR] No valid face RGB data collected.")
    exit()

# Interpolate missing RGB data per channel
rppg_rgb_frames = np.array([interpolate_nans(rppg_rgb_frames[:, i]) for i in range(3)])  # Shape: [3, frames]

# Expand dimension for POS algorithm input
rppg_input = np.expand_dims(rppg_rgb_frames, axis=0)  # Shape: [1, 3, frames]

# POS computation
H = cpu_POS(rppg_input, FPS)
pos_signal = H[0]  # shape: [frames]

# Post-processing
rppg_filtered = apply_bandpass_filter(pos_signal, 0.9, 2.4, FPS)      # rPPG (heart rate)
resp_filtered = apply_bandpass_filter(resp_raw, 0.1, 0.5, FPS)        # Respiratory

# Estimate BPM and BRPM
bpm = estimate_bpm(rppg_filtered, FPS, min_distance_sec=0.4)
if bpm:
    print(f"[INFO] Estimated Heart Rate: {bpm:.2f} BPM")
else:
    print("[WARN] Unable to estimate BPM.")

brpm_peaks = estimate_brpm(resp_filtered, FPS)

if brpm_peaks:
    print(f"[INFO] BRPM (Peaks): {brpm_peaks:.2f} breaths/min")
else:
    print("[WARN] Unable to estimate BRPM (Peaks).")

plot_signal(rppg_filtered, resp_filtered, FPS, bpm=bpm, brpm=brpm_peaks, output_dir="output", raw_pos=pos_signal)

