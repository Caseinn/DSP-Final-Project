# Tugas Besar Mata Kuliah Pengolahan Sinyal Digital (IF3024)
**Dosen Pengampu**: Martin Clinton Tosima Manullang, S.T., M.T.

## Deskripsi Proyek
![Thumbnail](https://i.imgur.com/gAJ2lQO.jpeg)
Proyek ini bertujuan untuk membangun sebuah sistem **pendeteksi sinyal fisiologis** berupa **sinyal denyut jantung (rPPG)** dan **sinyal respirasi (pernapasan)** secara **real-time** melalui **video webcam** tanpa kontak langsung (contactless). Sistem ini menggabungkan beberapa komponen utama:

- **Capture Video** menggunakan webcam
- **Ekstraksi sinyal rPPG** menggunakan algoritma POS (Plane-Orthogonal-to-Skin) berbasis deteksi wajah
- **Ekstraksi sinyal respirasi** dengan pelacakan pergerakan dada berdasarkan deteksi pose dari MediaPipe
- **Proses filtering sinyal** menggunakan bandpass filter untuk membersihkan noise
- **Estimasi HR (Heart Rate)** dan **RR (Respiration Rate)** secara real-time
- **Tampilan GUI interaktif** untuk memvisualisasikan sinyal dan grafik deteksi detak jantung serta napas

Seluruh komponen dibangun menggunakan **Python**, memanfaatkan library seperti **OpenCV**, **MediaPipe**, dan **PyQt5**, serta mengikuti arsitektur modular agar mudah dikembangkan.

## Struktur Proyek

```
DSP-Final-Project/
├── src/
│   ├── models/
│   │   └── pose_landmarker.task
│   ├── output/
│   ├── main.py
│   ├── signal_processing.py
│   ├── gui.py
│   └── utils.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Anggota Kelompok

| Nama                          | NIM        | ID GitHub            |
|-------------------------------|------------|----------------------|
| Dito Rifki Irawan              | 122140153  | [@Caseinn](https://github.com/Caseinn)  |
| Zefanya Danovanta Tarigan      | 122140101  | [@danovantaa](https://github.com/danovantaa) |

## Logbook

| Week | Progress |
|:----:|----------|
| 1    | - Mengintegrasikan pipeline video capture dan signal processing untuk ekstraksi rPPG<br>- Mengimplementasikan algoritma POS untuk ekstraksi sinyal rPPG<br>- Membuat face region tracking dengan MediaPipe<br>- Menambahkan signal processing dengan bandpass filtering<br>- Menyertakan visualisasi hasil heart rate |
| 2    | - Mengintegrasikan pipeline video capture dan signal processing untuk ekstraksi Sinyal Respirasi<br>- Menambahkan fitur **live signal monitoring** secara real-time<br>- Mengembangkan tampilan antarmuka pengguna (GUI) untuk visualisasi dan interaksi |
| 3    | - Melakukan **penyempurnaan antarmuka (GUI)** agar lebih responsif dan informatif<br>- Menambahkan **penanganan error** untuk memastikan program tidak crash saat webcam gagal atau sinyal tidak terbaca<br>- Melakukan **refactor kode** menjadi lebih modular dan bersih<br>- Menambahkan **docstring dan komentar** pada setiap fungsi dan modul untuk meningkatkan keterbacaan kode |

## Instalasi Program

Sebelum memulai proses instalasi, pastikan sistem komputer Anda telah memenuhi prasyarat minimum sebagai berikut:

### Prasyarat Sistem
- **Python**: Diperlukan versi Python 3.8 atau yang lebih tinggi untuk kompatibilitas penuh dengan dependensi dan lingkungan pengembangan
- **Webcam**: Diperlukan kamera webcam (terintegrasi atau eksternal) yang berfungsi dengan baik untuk akuisisi video secara real-time
- **Sistem Operasi**: Aplikasi ini dapat dijalankan pada berbagai sistem operasi populer, termasuk Windows, macOS, dan distribusi Linux

### 1. Kloning Repositori Proyek

Langkah pertama adalah mendapatkan kode sumber aplikasi. Buka terminal atau command prompt pada sistem Anda, lalu eksekusi perintah berikut untuk mengkloning repositori proyek dari GitHub dan masuk ke direktori proyek:

```bash
git clone https://github.com/Caseinn/DSP-Final-Project.git
cd DSP-Final-Project
```

Pastikan Anda memiliki git terinstal di sistem Anda. Jika belum, silakan instal terlebih dahulu.

### 2. Membuat dan Mengaktifkan Virtual Environment

Untuk menjaga lingkungan pengembangan tetap bersih dan terisolasi, buatlah virtual environment dengan perintah berikut:

```bash
python -m venv venv
```

Kemudian aktifkan environment tersebut sesuai sistem operasi Anda:

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Instalasi Dependensi

Setelah virtual environment aktif, instal semua pustaka dan paket Python yang diperlukan oleh aplikasi dengan perintah berikut:

```bash
pip install -r requirements.txt
```

**Rekomendasi**: Untuk proses instalasi yang lebih cepat, Anda juga dapat menggunakan `uv` jika telah terinstal:

```bash
uv pip install -r requirements.txt
```

Jika `uv` belum tersedia di sistem Anda, silakan merujuk pada dokumentasi resmi uv untuk petunjuk instalasi.

### 4. Menjalankan Aplikasi

Setelah semua dependensi berhasil diinstal, navigasikan ke direktori `src` dan jalankan aplikasi dengan perintah berikut:

```bash
cd src
python main.py
```

Aplikasi akan mulai berjalan dan Anda dapat mulai menggunakan fitur-fitur yang tersedia dalam proyek ini.

## Referensi

[1] A. Goel, N. Gupta, A. Zehra, V. Raj, and A. Malik, "Real time heart rate monitoring using web-camera," International Journal of Advanced Research (IJAR), vol. 11, no. 4, pp. 1264–1277, April 2023. [Online]. Available: https://dx.doi.org/10.21474/IJAR01/16789

[2] W. Wang, A. C. den Brinker, S. Stuijk, and G. de Haan, "Algorithmic principles of remote ppg," in IEEE Transactions on Biomedical Engineering, vol. 64, no. 7, 2017, pp. 1479–1491.

[3] D. Pagliari and L. Pinto, "Contactless physiological monitoring from videos: A review," Computer Vision and Image Understanding, vol. 169, pp. 102–119, 2018.

[4] T.-H. Pham, T. K. D. Vu, and Y.-K. Lee, "Deep learning-based non-contact vital sign monitoring: A review," IEEE Access, vol. 10, pp. 71 530–71 552, 2022.

[5] D.-Y. Kim, K. Lee, and C.-B. Sohn, "Assessment of roi selection for facial video-based rppg," Sensors, vol. 21, no. 23, 2021. [Online]. Available: https://www.mdpi.com/1424-8220/21/23/7923

---

**Program Pendeteksi Sinyal Respirasi dan Sinyal rPPG**  
Dito Rifki Irawan (122140153), Zefanya Danovanta Tarigan (122140101)
