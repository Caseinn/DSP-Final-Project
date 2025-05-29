import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def cpu_POS(signal, fps):
    """
    Menghitung sinyal Photoplethysmographic (rPPG) menggunakan algoritma POS.
    
    Args:
        signal (np.ndarray): Sinyal input dari ROI wajah dengan bentuk [estimator, channel warna, frame]
        fps (int): Frame per detik (frekuensi sampling)
        
    Returns:
        np.ndarray: Sinyal rPPG hasil pemrosesan POS
    """
    eps = 1e-9
    X = signal  # shape: [estimators, 3, frames]
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

def butter_bandpass(lowcut, highcut, fs, order=3):
    """
    Membuat filter bandpass Butterworth untuk frekuensi tertentu.
    
    Args:
        lowcut (float): Frekuensi batas bawah (Hz)
        highcut (float): Frekuensi batas atas (Hz)
        fs (int): Frekuensi sampling (Hz)
        order (int): Orde filter
        
    Returns:
        tuple: Koefisien numerik dan denominatif dari filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def apply_bandpass_filter(data, lowcut, highcut, fs):
    """
    Menerapkan filter bandpass pada sinyal menggunakan koefisien dari butter_bandpass.
    
    Args:
        data (np.ndarray): Sinyal masukan
        lowcut (float): Frekuensi batas bawah (Hz)
        highcut (float): Frekuensi batas atas (Hz)
        fs (int): Frekuensi sampling (Hz)
        
    Returns:
        np.ndarray: Sinyal yang telah difilter
    """
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)

def estimate_bpm(signal, fps, min_distance_sec=0.5):
    """
    Mengestimasi denyut jantung (BPM) dari sinyal rPPG menggunakan deteksi puncak.
    
    Args:
        signal (np.ndarray): Sinyal rPPG yang telah difilter
        fps (int): Frame per detik (frekuensi sampling)
        min_distance_sec (float): Jarak minimum antar puncak dalam detik (default: 0.5 detik)
        
    Returns:
        float or None: Estimasi BPM atau None jika tidak cukup puncak ditemukan
    """
    min_distance = int(min_distance_sec * fps)

    peaks, _ = find_peaks(signal, distance=min_distance)

    if len(peaks) < 2:
        return None

    duration_min = len(signal) / fps / 60.0 

    bpm = len(peaks) / duration_min
    return bpm

def estimate_brpm(signal, fps, min_distance_sec=1.5):
    """
    Mengestimasi laju pernapasan (BRPM) dari sinyal pernapasan menggunakan deteksi puncak.
    
    Args:
        signal (np.ndarray): Sinyal pernapasan yang telah difilter
        fps (int): Frame per detik (frekuensi sampling)
        min_distance_sec (float): Jarak minimum antar puncak dalam detik (default: 1.5 detik)
        
    Returns:
        float or None: Estimasi BRPM atau None jika tidak cukup puncak ditemukan
    """
    distance = int(min_distance_sec * fps)
    peaks, _ = find_peaks(signal, distance=distance)
    duration_min = len(signal) / fps / 60.0
    brpm = len(peaks) / duration_min if duration_min > 0 else None
    return brpm