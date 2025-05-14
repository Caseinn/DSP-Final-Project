# Konfigurasi sistem rPPG 
CONFIG = {
    "FPS": 30,  # Frame per detik
    "MAX_FRAMES": 600,  # Jumlah maksimum frame yang akan diproses
    "COUNTDOWN_SECONDS": 3,  # Hitung mundur sebelum merekam
    "FILTER_PARAMS": {
        # Rentang frekuensi untuk deteksi denyut jantung (0.9–2.4 Hz)
        "rppg_band": {"low": 0.9, "high": 2.4},
    },
    "OUTPUT_DIR": "result"  # Direktori hasil output
}