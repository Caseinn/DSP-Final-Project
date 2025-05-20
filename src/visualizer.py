import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer
import sys
import mediapipe as mp
from utils import FACE_REGIONS, draw_face_roi, draw_shoulders, extract_face_roi_rgb, extract_shoulder_distance
import csv
from countdown import start_countdown
from collections import deque
import pyqtgraph as pg
from signal_processing import cpu_POS, apply_bandpass_filter, estimate_bpm, estimate_brpm

            
def convert_cv_qt(cv_img):
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

class VideoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("rPPG + Respirasi GUI")
        self.image_label = QLabel()
        self.start_btn = QPushButton("Start Capture")
        self.stop_btn = QPushButton("Stop & Save Data")
        self.stop_btn.clicked.connect(self.stop_camera_and_save)
        self.plot_widget = pg.PlotWidget(title="Live rPPG Signal")
        self.plot_widget.setYRange(-0.1, 0.1)  # sesuaikan rentang
        self.bpm_label = QLabel("BPM: -")
        self.brpm_label = QLabel("BRPM: -")


        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.plot_widget)
        layout.addWidget(self.bpm_label)
        layout.addWidget(self.brpm_label)
        self.setLayout(layout)

        self.start_btn.clicked.connect(self.start_camera)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Inisialisasi Mediapipe
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.pose = mp.solutions.pose.Pose()
        self.rgb_data = []      
        self.resp_data = []  
        self.rppg_buffer = deque(maxlen=150)  # 5 detik @30 FPS
        self.resp_buffer = deque(maxlen=150)

        self.analysis_timer = QTimer()
        self.analysis_timer.timeout.connect(self.analyze_signals)
        self.analysis_timer.start(3000)  # analisis setiap 3 detik


    def start_camera(self):
        # start_countdown(3)
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)  # roughly 30 FPS

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_result = self.face_mesh.process(rgb)
        pose_result = self.pose.process(rgb)

        # Deteksi wajah dan gambar ROI
        if face_result.multi_face_landmarks:
            landmarks = face_result.multi_face_landmarks[0].landmark
            draw_face_roi(frame, landmarks, FACE_REGIONS)

            roi_rgb = []
            for region in FACE_REGIONS.values():
                rgb_val = extract_face_roi_rgb(frame, landmarks, region)
                roi_rgb.append(rgb_val)
            avg_rgb = np.mean(roi_rgb, axis=0)
            self.rgb_data.append(avg_rgb.tolist())
            self.rppg_buffer.append(avg_rgb.tolist())  # ✅ Tambahkan baris ini
        else:
            self.rgb_data.append([np.nan, np.nan, np.nan])


        # Deteksi pose dan gambar bahu
        if pose_result.pose_landmarks:
            draw_shoulders(frame, pose_result.pose_landmarks.landmark)
            distance = extract_shoulder_distance(pose_result.pose_landmarks.landmark)
            self.resp_data.append(distance if distance is not None else np.nan)
            self.resp_buffer.append(distance if distance is not None else np.nan)  # ✅ Tambahkan baris ini
        else:
            self.resp_data.append(np.nan)


        qt_img = convert_cv_qt(frame)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))
    
    def analyze_signals(self):
        if len(self.rppg_buffer) < 60:  # minimal data
            return

        fps = 30
        rgb = np.array(self.rppg_buffer).T  # shape: [3, frames]
        rgb = np.expand_dims(rgb, axis=0)   # shape: [1, 3, frames]

        H = cpu_POS(rgb, fps)
        pos_signal = H[0]

        rppg_filtered = apply_bandpass_filter(pos_signal, 0.9, 2.4, fps)
        bpm = estimate_bpm(rppg_filtered, fps)

        # Plot rPPG
        self.plot_widget.clear()
        self.plot_widget.plot(rppg_filtered[-150:], pen='g')

        # Update label BPM
        if bpm:
            self.bpm_label.setText(f"BPM: {bpm:.2f}")
        else:
            self.bpm_label.setText("BPM: -")

        # Respirasi
        resp_array = np.array(self.resp_buffer)
        if np.isnan(resp_array).all():
            self.brpm_label.setText("BRPM: -")
            return

        resp_filtered = apply_bandpass_filter(resp_array, 0.1, 0.5, fps)
        brpm = estimate_brpm(resp_filtered, fps)

        if brpm:
            self.brpm_label.setText(f"BRPM: {brpm:.2f}")
        else:
            self.brpm_label.setText("BRPM: -")

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()
        
    def stop_camera_and_save(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.timer.stop()

        # Simpan data ke CSV
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rgb_filename = f"output/rppg_rgb_{timestamp}.csv"
        resp_filename = f"output/resp_signal_{timestamp}.csv"

        # Simpan RGB data
        with open(rgb_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["R", "G", "B"])
            writer.writerows(self.rgb_data)
        print(f"[INFO] Saved RGB data to {rgb_filename}")

        # Simpan Respirasi data
        with open(resp_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Shoulder_Distance"])
            for d in self.resp_data:
                writer.writerow([d])
        print(f"[INFO] Saved respiratory data to {resp_filename}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoApp()
    win.show()
    sys.exit(app.exec_())