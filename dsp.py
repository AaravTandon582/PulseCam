"""Pure DSP: (timestamps, values) -> BPM. numpy/scipy only — no cv2, no I/O.

Pipeline: measured fs -> resample to uniform grid -> moving-average detrend ->
Butterworth bandpass 0.7-4 Hz -> Hann + zero-padded rFFT -> in-band peak with
parabolic interpolation -> BPM + SNR. Testable without a camera (test_dsp.py).
"""
import numpy as np
from scipy.signal import butter, filtfilt

HR_BAND = (0.7, 4.0)   # Hz == 42-240 BPM


def measured_fs(t):
    dt = np.median(np.diff(t))
    return 1.0 / dt if dt > 0 else 0.0


def resample_uniform(t, v, fs):
    n = int(round((t[-1] - t[0]) * fs)) + 1
    tu = t[0] + np.arange(n) / fs
    return tu, np.interp(tu, t, v)


def detrend(v, fs, window_s=1.5):
    w = max(3, int(window_s * fs) | 1)
    baseline = np.convolve(np.pad(v, w // 2, mode="edge"),
                           np.ones(w) / w, mode="valid")
    return v - baseline


def bandpass(v, fs, lo=HR_BAND[0], hi=HR_BAND[1], order=3):
    nyq = fs / 2.0
    hi = min(hi, 0.95 * nyq)   # stay below Nyquist if the camera runs slow
    if hi <= lo:
        return v
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, v)


def spectrum(v, fs, pad_factor=4):
    n = len(v)
    nfft = int(2 ** np.ceil(np.log2(n * pad_factor)))
    power = np.abs(np.fft.rfft(v * np.hanning(n), nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    return freqs, power


def pick_peak(freqs, power, lo=HR_BAND[0], hi=HR_BAND[1]):
    """Dominant in-band peak -> (f_peak, snr); (None, 0) if band is empty."""
    band = (freqs >= lo) & (freqs <= hi)
    if band.sum() < 3:
        return None, 0.0
    idx = np.flatnonzero(band)
    i = idx[np.argmax(power[idx])]

    f_peak = freqs[i]
    if 0 < i < len(power) - 1:
        y0, y1, y2 = power[i - 1], power[i], power[i + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            f_peak += 0.5 * (y0 - y2) / denom * (freqs[1] - freqs[0])

    # SNR: peak power vs mean in-band power away from the peak
    near = band & (np.abs(freqs - f_peak) <= 0.2)
    rest = band & ~near
    noise = power[rest].mean() if rest.any() else power[band].mean()
    snr = float(power[i] / noise) if noise > 0 else 0.0
    return float(f_peak), snr


def estimate(t, v, min_seconds=5.0):
    """Full pipeline over one window. Returns a dict (bpm, snr, fs, and every
    intermediate signal for the instrumentation panels), or None if the data
    can't support an estimate yet."""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    if len(t) < 20 or t[-1] - t[0] < min_seconds:
        return None
    fs = measured_fs(t)
    if fs < 5.0:   # below this the HR band is mostly gone
        return None
    tu, raw = resample_uniform(t, v, fs)
    filtered = bandpass(detrend(raw, fs), fs)
    freqs, power = spectrum(filtered, fs)
    f_peak, snr = pick_peak(freqs, power)
    if f_peak is None:
        return None
    return {"bpm": 60.0 * f_peak, "snr": snr, "fs": fs,
            "t": tu, "raw": raw, "filtered": filtered,
            "freqs": freqs, "power": power}
