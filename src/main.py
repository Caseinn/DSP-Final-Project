from PyQt5.QtWidgets import QApplication
from gui import VideoApp
import sys

if __name__ == "__main__":
    """
    Aplikasi utama untuk pengambilan data rPPG dan respirasi.
    Menggunakan MediaPipe untuk mendeteksi wajah dan pose.
    Menjalankan countdown sebelum mulai, menyimpan video dan data,
    lalu menganalisisnya untuk mendapatkan BPM dan BRPM.
    """
    app = QApplication(sys.argv)
    win = VideoApp()
    win.show()
    sys.exit(app.exec_())