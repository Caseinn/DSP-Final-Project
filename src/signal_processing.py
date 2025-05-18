import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def cpu_POS(signal, fps):
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
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def apply_bandpass_filter(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)

def estimate_bpm(signal, fps, min_distance_sec=0.5):
    """
    Estimate heart rate (BPM) from the signal using peak detection.

    Args:
        signal (np.ndarray): 1D filtered rPPG signal
        fps (int): Frames per second (sampling rate)
        min_distance_sec (float): Minimum time between heartbeats in seconds. 
                                  Default is 0.5s (for up to 120 BPM)

    Returns:
        float or None: Estimated BPM or None if no peaks detected
    """
    # Minimum distance between peaks in samples
    min_distance = int(min_distance_sec * fps)

    # Detect peaks (use a dynamic threshold if needed)
    peaks, _ = find_peaks(signal, distance=min_distance)

    if len(peaks) < 2:
        return None  # Not enough peaks to estimate BPM

    # Compute duration of the signal in minutes
    duration_min = len(signal) / fps / 60.0  # seconds → minutes

    # Estimate BPM from peak count
    bpm = len(peaks) / duration_min
    return bpm