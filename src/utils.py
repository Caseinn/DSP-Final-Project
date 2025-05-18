# src/utils.py

import matplotlib.pyplot as plt
import time
import numpy as np
from datetime import datetime
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2  

FACE_REGIONS = {
    "right_cheek": [36, 50, 101, 118, 117, 116, 123, 147, 187, 205],
    "left_cheek": [266, 330, 347, 346, 345, 352, 376, 411, 425, 280],
    "forehead": [10, 109, 338, 108, 107 , 9, 336, 337, 151]
}

def extract_face_roi_rgb(frame, landmarks, region_ids):
    h, w, _ = frame.shape
    pixels = []
    for idx in region_ids:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        pixels.append(frame[y, x])
    return np.mean(pixels, axis=0)  # R, G, B average

def plot_signal(rppg_signal, fps, bpm=None, output_dir="output", raw_pos=None):
    """
    Plot rPPG and respiration signals in subplots and save with timestamp.
    Optionally overlays raw POS signal.
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{timestamp}_signals.png"

    t = np.arange(len(rppg_signal)) / fps

    fig, axs = plt.subplots(2 if raw_pos is not None else 1, 1, figsize=(14, 10))

    axs[0].plot(t, raw_pos, color='gray', label="Raw POS")
    axs[0].set_title("Raw POS Signal (Before Filtering)")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)

    axs[1].plot(t, rppg_signal, color='green', label="Filtered POS")
    axs[1].plot(t, raw_pos, color='gray', linestyle='--', alpha=0.6, label="Raw POS")
    axs[1].set_title(f"rPPG Signal - Estimated Heart Rate: {bpm:.2f} BPM" if bpm else "rPPG Signal")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()
    axs[1].grid(True)

    fig.suptitle(f"Physiological Signals - {timestamp}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    plt.close()
    print(f"[INFO] Plot saved as {filename}")


def interpolate_nans(signal):
    """
    Interpolate NaN values in a 1D or 2D array using linear interpolation.

    Args:
        signal (np.ndarray): Input signal with NaNs.

    Returns:
        np.ndarray: Signal with NaNs replaced via interpolation.
    """
    if signal.ndim == 1:
        nans = np.isnan(signal)
        if not np.any(nans):
            return signal
        x = ~nans
        xp = np.where(x)[0]
        fp = signal[x]
        interp = np.interp(np.where(nans)[0], xp, fp)
        result = signal.copy()
        result[nans] = interp
        return result
    elif signal.ndim == 2:
        return np.vstack([interpolate_nans(row) for row in signal])
    else:
        raise ValueError("Signal must be 1D or 2D")
    
def plot_to_image(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()
    image = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))  # RGBA
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return image_rgb
