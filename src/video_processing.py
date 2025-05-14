# src/video_processing.py
import cv2
import time
import numpy as np
import os
import mediapipe as mp
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from config import CONFIG
from signal_processing import cpu_POS, bandpass_filter, lowpass_filter
from signal_extraction import extract_rppg_roi

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

def setup_video_capture():
    """Inisialisasi webcam dengan penanganan error"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to access webcam")
    cap.set(cv2.CAP_PROP_FPS, CONFIG["FPS"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def visualize_tracking(frame, face_points):
    """Gambar elemen visualisasi pada frame"""
    if face_points:
        for point in face_points:
            cv2.circle(frame, point, 3, (255, 0, 0), -1)
            roi_size = 10
            cv2.rectangle(
                frame,
                (point[0] - roi_size, point[1] - roi_size),
                (point[0] + roi_size, point[1] + roi_size),
                (255, 0, 0),
                1
            )

def process_frame(frame):
    """Proses frame untuk ekstraksi sinyal dan visualisasi"""
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_result = face_mesh.process(rgb_frame)
        
        rppg_rgb, face_points = extract_rppg_roi(rgb_frame, face_result)

        visualize_tracking(frame, face_points)

        return rppg_rgb 

    except Exception as e:
        print(f"Frame processing error: {e}")
        return None

def generate_visualization(rppg_results, frames_recorded=0):
    """Buat dan simpan plot visualisasi sinyal"""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    time_axis = np.arange(len(rppg_results['filtered'])) / CONFIG["FPS"]
    ax.plot(time_axis, rppg_results['filtered'], label="Filtered rPPG")
    ax.plot(time_axis[rppg_results['peaks']], rppg_results['filtered'][rppg_results['peaks']], 
           "rx", label="Heart Peaks")
    ax.set_title(f"Heart Rate: {rppg_results['heart_rate']:.1f} BPM")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(CONFIG["OUTPUT_DIR"], f"signals_{timestamp}.png")
    fig.savefig(plot_path)
    print(f"Results saved to {plot_path}")
    
    recorded_time = frames_recorded / CONFIG["FPS"]
    print(f"Recording duration: {recorded_time:.1f} seconds ({frames_recorded} frames at {CONFIG['FPS']} FPS)")
    print(f"Heart Rate: {rppg_results['heart_rate']:.1f} BPM")
    
    plt.show()
    plt.close(fig)

def record_rppg():
    cap = setup_video_capture()
    try:
        # Countdown
        for i in range(CONFIG["COUNTDOWN_SECONDS"], 0, -1):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame during countdown")
            h, w = frame.shape[:2]
            cv2.putText(
                frame, str(i),
                (w//2 - 20, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5
            )
            cv2.imshow('Countdown', frame)
            cv2.waitKey(1000)
        cv2.destroyWindow('Countdown')
        
        # Data collection
        frame_count = 0
        rppg_data = []
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
        while frame_count < CONFIG["MAX_FRAMES"]:
            ret, frame = cap.read()
            if not ret:
                break
                
            rppg_rgb = process_frame(frame)

            if rppg_rgb is not None:
                rppg_data.append(rppg_rgb)
                
            # Display real-time info
            cv2.putText(
                frame,  
                f"Frame: {frame_count}/{CONFIG['MAX_FRAMES']} | FPS: {fps:.1f}",  
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 255, 255), 
                1 
            )

            cv2.imshow('rPPG Capture', frame)
            if cv2.waitKey(1) == ord('q'):
                break
                
            # FPS calculation
            fps_counter += 1
            if (time.time() - fps_start_time) > 1:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
                
            frame_count += 1
            
        # Process and save results
        if not os.path.exists(CONFIG["OUTPUT_DIR"]):
            os.makedirs(CONFIG["OUTPUT_DIR"])
            
        if len(rppg_data) >= 3:
            rppg_array = np.array(rppg_data).T
            rppg_array = rppg_array.reshape(1, 3, -1)
            
            rppg_raw = cpu_POS(rppg_array, CONFIG["FPS"])[0]
            filtered_rppg = bandpass_filter(
                rppg_raw,
                CONFIG["FILTER_PARAMS"]["rppg_band"]["low"],
                CONFIG["FILTER_PARAMS"]["rppg_band"]["high"],
                CONFIG["FPS"]
            )
            
            peaks, _ = find_peaks(filtered_rppg, height=np.mean(filtered_rppg) + 0.5 * np.std(filtered_rppg))
            heart_rate = 60 * len(peaks) / (len(filtered_rppg) / CONFIG["FPS"])
            
            # Generate visualization
            generate_visualization(
                {'filtered': filtered_rppg, 'peaks': peaks, 'heart_rate': heart_rate},  
            )
            
    except Exception as e:
        print(f"Runtime error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()