import cv2
import numpy as np
import csv
import os
from datetime import datetime
from collections import deque

from PyQt5.QtWidgets import (
    QLabel, QPushButton, QWidget,
    QVBoxLayout, QHBoxLayout, QMessageBox, QFrame
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt

import pyqtgraph as pg
import mediapipe as mp

from utils import FACE_REGIONS, draw_face_roi, draw_shoulders, extract_face_roi_rgb, extract_shoulder_distance
from signal_processing import cpu_POS, apply_bandpass_filter, estimate_bpm, estimate_brpm

def convert_cv_qt(cv_img):
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

class VideoApp(QWidget):
    def __init__(self):
        """
        Inisialisasi GUI aplikasi rPPG + Respirasi.
        Menyiapkan komponen seperti label gambar, tombol kontrol, grafik, dan timer.
        """
        super().__init__()
        self.setWindowTitle("rPPG + Respirasi GUI")
        self.setFixedSize(1280, 720)
        os.makedirs("output", exist_ok=True)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        placeholder = QPixmap(640, 480)
        placeholder.fill(Qt.black)
        self.image_label.setPixmap(placeholder)

        self.start_btn = QPushButton("Start Capture")
        self.stop_btn = QPushButton("Stop && Save Data")
        self.stop_btn.setEnabled(False)

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera_and_save)

        self.plot_widget = pg.PlotWidget(title="Live rPPG Signal")
        self.plot_widget.setMinimumHeight(200)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.setLabel('bottom', 'Time (s)')

        self.plot_widget_resp = pg.PlotWidget(title="Live Respirasi Signal")
        self.plot_widget_resp.setMinimumHeight(200)
        self.plot_widget_resp.showGrid(x=True, y=True)
        self.plot_widget_resp.setLabel('left', 'Distance')
        self.plot_widget_resp.setLabel('bottom', 'Time (s)')

        font = QFont("Arial", 11, QFont.Bold)
        self.status_label = QLabel("Status: Waiting to start...")
        self.status_label.setFont(font)

        self.duration_label = QLabel("Duration: 0 s")
        self.duration_label.setFont(font)

        self.bpm_card = QLabel("BPM: -")
        self.bpm_card.setFrameShape(QFrame.Box)
        self.bpm_card.setStyleSheet("background-color: #e0f7fa; font-size: 16px; padding: 8px;")
        self.bpm_card.setAlignment(Qt.AlignCenter)

        self.brpm_card = QLabel("BRPM: -")
        self.brpm_card.setFrameShape(QFrame.Box)
        self.brpm_card.setStyleSheet("background-color: #fce4ec; font-size: 16px; padding: 8px;")
        self.brpm_card.setAlignment(Qt.AlignCenter)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.image_label)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.plot_widget)
        right_layout.addWidget(self.plot_widget_resp)

        top_layout = QHBoxLayout()
        top_layout.addLayout(left_layout, 1)
        top_layout.addLayout(right_layout, 1)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.setAlignment(Qt.AlignCenter)

        info_layout = QVBoxLayout()
        info_layout.addWidget(self.status_label)
        info_layout.addWidget(self.duration_label)
        info_layout.addWidget(self.bpm_card)
        info_layout.addWidget(self.brpm_card)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(button_layout)
        main_layout.addLayout(info_layout)

        self.setLayout(main_layout)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.analysis_timer = QTimer()
        self.analysis_timer.timeout.connect(self.analyze_signals)

        self.duration_timer = QTimer()
        self.duration_timer.timeout.connect(self.update_duration)
        self.capture_start_time = None

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.pose = mp.solutions.pose.Pose()

        self.rgb_data = []
        self.resp_data = []
        self.rppg_buffer = deque(maxlen=150)
        self.resp_buffer = deque(maxlen=150)

    def update_duration(self):
        """
        Memperbarui label durasi selama pengambilan data berlangsung.
        """
        if self.capture_start_time:
            elapsed = int((datetime.now() - self.capture_start_time).total_seconds())
            self.duration_label.setText(f"Duration: {elapsed} s")

    def start_camera(self):
        """
        Memulai kamera dan memulai pengambilan data serta analisis sinyal.
        Membersihkan buffer dan mereset status GUI sebelum mulai.
        """
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Tidak dapat membuka kamera.")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.timer.start(30)

        self.rgb_data.clear()
        self.resp_data.clear()
        self.rppg_buffer.clear()
        self.resp_buffer.clear()

        self.plot_widget.clear()
        self.plot_widget_resp.clear()
        self.bpm_card.setText("BPM: -")
        self.brpm_card.setText("BRPM: -")
        self.duration_label.setText("Duration: 0 s")

        self.capture_start_time = datetime.now()
        self.duration_timer.start(1000)
        self.analysis_timer.start(3000)

        self.status_label.setText("Status: Capturing...")
        self.stop_btn.setEnabled(True)
        self.start_btn.setEnabled(False)

    def update_frame(self):
        """
        Memperbarui frame video secara real-time.
        Deteksi ROI wajah dan bahu, hitung data rPPG & respirasi, dan update GUI.
        """
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_result = self.face_mesh.process(rgb)
        if face_result.multi_face_landmarks:
            landmarks = face_result.multi_face_landmarks[0].landmark
            draw_face_roi(frame, landmarks, FACE_REGIONS)
            roi_rgb = [extract_face_roi_rgb(frame, landmarks, region) for region in FACE_REGIONS.values()]
            avg_rgb = np.mean(roi_rgb, axis=0)
            self.rgb_data.append(avg_rgb.tolist())
            self.rppg_buffer.append(avg_rgb.tolist())
            self.status_label.setText("Status: Wajah terdeteksi")
        else:
            self.rgb_data.append([np.nan, np.nan, np.nan])
            self.rppg_buffer.append([np.nan, np.nan, np.nan])
            self.status_label.setText("Status: Wajah tidak terdeteksi")

        pose_result = self.pose.process(rgb)
        if pose_result.pose_landmarks:
            draw_shoulders(frame, pose_result.pose_landmarks.landmark)
            distance = extract_shoulder_distance(pose_result.pose_landmarks.landmark, min_visibility=0.3)
            if distance is not None:
                self.resp_data.append(distance)
                self.resp_buffer.append(distance)
            else:
                self.resp_data.append(np.nan)
                self.resp_buffer.append(np.nan)
                self.status_label.setText("Status: Bahu terdeteksi, tapi visibilitas rendah")
        else:
            self.resp_data.append(np.nan)
            self.resp_buffer.append(np.nan)
            self.status_label.setText("Status: Landmark pose tidak terdeteksi")

        qt_img = convert_cv_qt(frame)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def analyze_signals(self):
        """
        Menganalisis sinyal rPPG dan respirasi setiap beberapa detik.
        Menghitung BPM dan BRPM serta memperbarui plot dan label GUI.
        """
        if len(self.rppg_buffer) < 60:
            return

        fps = 30
        rgb = np.array(self.rppg_buffer).T
        rgb = np.expand_dims(rgb, axis=0)
        H = cpu_POS(rgb, fps)
        pos_signal = H[0]

        rppg_filtered = apply_bandpass_filter(pos_signal, 0.9, 2.4, fps)
        bpm = estimate_bpm(rppg_filtered, fps)

        self.plot_widget.clear()
        self.plot_widget.plot(rppg_filtered[-150:], pen='g')
        self.bpm_card.setText(f"BPM: {bpm:.2f}" if bpm else "BPM: -")

        resp_array = np.array(self.resp_buffer)
        if not np.isnan(resp_array).all():
            resp_filtered = apply_bandpass_filter(resp_array, 0.1, 0.5, fps)
            brpm = estimate_brpm(resp_filtered, fps)
            self.plot_widget_resp.clear()
            self.plot_widget_resp.plot(resp_filtered[-150:], pen='b')
            self.brpm_card.setText(f"BRPM: {brpm:.2f}" if brpm else "BRPM: -")
        else:
            self.brpm_card.setText("BRPM: -")

    def stop_camera_and_save(self):
        """
        Menghentikan kamera dan menyimpan data rPPG serta respirasi ke file CSV.
        Mereset tampilan dan status GUI setelah penyimpanan selesai.
        """
        if self.cap:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        self.analysis_timer.stop()
        self.duration_timer.stop()
        self.duration_label.setText("Duration: 0 s")
        self.capture_start_time = None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rgb_filename = f"output/rppg_rgb_{timestamp}.csv"
        resp_filename = f"output/resp_signal_{timestamp}.csv"

        try:
            with open(rgb_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["R", "G", "B"])
                writer.writerows(self.rgb_data)

            with open(resp_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Shoulder_Distance"])
                for d in self.resp_data:
                    writer.writerow([d])

            QMessageBox.information(self, "Success", "Data has been saved successfully.")
            self.status_label.setText("Status: Data saved.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save data: {str(e)}")
            self.status_label.setText("Status: Error saving data.")

        placeholder = QPixmap(640, 480)
        placeholder.fill(Qt.black)
        self.image_label.setPixmap(placeholder)
        self.plot_widget.clear()
        self.plot_widget_resp.clear()
        self.bpm_card.setText("BPM: -")
        self.brpm_card.setText("BRPM: -")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def closeEvent(self, event):
        """
        Menangani penutupan aplikasi, melepaskan resource kamera sebelum keluar.
        
        Args:
            event (QCloseEvent): Event penutupan jendela
        """
        if self.cap:
            self.cap.release()
        event.accept()
