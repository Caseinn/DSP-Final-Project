# src/signal_processing.py
"""
Modul pemrosesan sinyal untuk ekstraksi sinyal rPPG dan pernapasan.
"""
import numpy as np
import scipy.signal as signal

def cpu_POS(signal, fps):
    """
    Process sinyal photoplethysmography (rPPG) menggunakan algoritma POS.
    Parameters:
        signal (np.ndarray): Input RGB signal (bentuk: [estimators, 3, frames])
        fps (int): Frames per second dari video
    Returns:
        np.ndarray: Sinyal rPPG yang diproses (bentuk: [estimators, frames])
    """
    eps = 1e-9
    X = signal
    e, c, f = X.shape
    w = int(1.6 * fps)
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)
    H = np.zeros((e, f))
    for n in range(w, f):
        m = n - w + 1
        Cn = X[:, :, m:n+1]
        M = 1.0 / (np.mean(Cn, axis=2) + eps)
        M = np.expand_dims(M, axis=2)
        Cn = np.multiply(M, Cn)
        S = np.dot(Q, Cn)
        S = S[0, :, :, :]
        S = np.swapaxes(S, 0, 1)
        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha = np.expand_dims(alpha, axis=1)
        Hn = np.add(S1, alpha * S2)
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
        H[:, m:n+1] = np.add(H[:, m:n+1], Hnm)
    return H

def bandpass_filter(data, lowcut=0.9, highcut=2.4, fs=30, order=5):
    """
    Terapkan Butterworth bandpass filter untuk pemrosesan sinyal rPPG.
    Dirancang untuk deteksi denyut jantung:
    - Rentang denyut jantung manusia: 0.7–2.5 Hz (42–150 BPM)
    - Rentang terpilih (0.9–2.4 Hz) menghindari gerakan (<0.7 Hz) 
      dan noise pengukuran (>2.5 Hz)
    - Zero-phase filtering menjaga bentuk gelombang
    Parameters:
        data (np.ndarray): Input signal
        lowcut (float): Frekuensi batas bawah (Hz)
        highcut (float): Frekuensi batas atas (Hz)
        fs (int): Frekuensi sampling (Hz)
        order (int): Orde filter (dikurangi dari default 5 untuk meminimalkan delay)
    Returns:
        np.ndarray: Sinyal yang difilter dengan komponen fisiologis
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def lowpass_filter(data, cutoff=0.5, fs=30, order=3):
    """
    Terapkan Butterworth low-pass filter untuk pemrosesan sinyal pernapasan.
    Justifikasi laju pernapasan:
    - Laju pernapasan dewasa normal: 0.1–0.5 Hz (6–30 napas/menit)
    - Cut-off 0.5 Hz mempertahankan semua laju pernapasan fisiologis
    - Frekuensi lebih tinggi merepresentasikan gerakan tubuh
    Parameters:
        data (np.ndarray): Input signal
        cutoff (float): Frekuensi cut-off (Hz)
        fs (int): Frekuensi sampling (Hz)
        order (int): Orde filter (3 memberikan roll-off yang cukup)
    Returns:
        np.ndarray: Sinyal pernapasan yang difilter
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low')
    return signal.filtfilt(b, a, data)