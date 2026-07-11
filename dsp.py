"""Pure DSP: (timestamps, channels) -> BPM. numpy/scipy only — no cv2, no I/O.

Pipeline: measured fs -> resample to uniform grid -> pulse extraction (POS on
RGB, or detrended green) -> Butterworth bandpass 0.7-3 Hz -> Hann + zero-padded
rFFT -> in-band peak with parabolic interpolation -> BPM + SNR. Testable
without a camera (test_dsp.py).
"""
import numpy as np
from scipy.signal import butter, filtfilt

HR_BAND = (0.7, 3.0)   # Hz == 42-180 BPM


def measured_fs(t):
    dt = np.median(np.diff(t))
    return 1.0 / dt if dt > 0 else 0.0


def uniform_grid(t, fs):
    n = int(round((t[-1] - t[0]) * fs)) + 1
    return t[0] + np.arange(n) / fs


def resample_uniform(t, v, fs):
    tu = uniform_grid(t, fs)
    return tu, np.interp(tu, t, v)


def pos_signal(rgb, fs, window_s=1.6):
    """POS — Plane-Orthogonal-to-Skin (Wang et al. 2017). Turns an (N, 3) RGB
    trace into a single robust pulse signal of length N.

    Per sliding window it temporally normalizes each channel by its own mean,
    projects onto the two directions orthogonal to the skin-tone axis
    (P = [[0,1,-1], [-2,1,1]]), alpha-tunes the two projections by their std
    ratio, and overlap-adds. Intensity changes common to all channels (motion,
    specular glare, flicker) land on the skin-tone axis and cancel in the
    projection; the pulse, which has a channel-specific signature, survives.
    """
    rgb = np.asarray(rgb, float)
    n = rgb.shape[0]
    l = max(2, int(round(window_s * fs)))
    c = rgb.T                                   # 3 x N
    p_mat = np.array([[0., 1., -1.], [-2., 1., 1.]])
    h = np.zeros(n)
    for end in range(l - 1, n):
        start = end - l + 1
        block = c[:, start:end + 1]             # 3 x l
        mu = block.mean(axis=1, keepdims=True)
        mu[mu == 0] = 1e-9
        cn = block / mu                         # temporal normalization
        s = p_mat @ cn                          # 2 x l projections
        std1 = s[1].std()
        alpha = s[0].std() / std1 if std1 > 1e-9 else 0.0
        window = s[0] + alpha * s[1]
        h[start:end + 1] += window - window.mean()
    return h


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


def peak_dominance(freqs, power, f_peak, lo=HR_BAND[0], hi=HR_BAND[1]):
    """Peak power divided by the strongest well-separated in-band competitor."""
    band = (freqs >= lo) & (freqs <= hi)
    competitors = band & (np.abs(freqs - f_peak) > 0.2)
    if not competitors.any():
        return float("inf")
    peak_power = power[np.argmin(np.abs(freqs - f_peak))]
    competitor = power[competitors].max()
    return float(peak_power / competitor) if competitor > 0 else float("inf")


def estimate(t, channels, method="pos", min_seconds=5.0, hr_band=HR_BAND):
    """Full pipeline over one window -> dict (bpm, snr, fs, method, and every
    intermediate signal for the instrumentation panels), or None if the data
    can't support an estimate yet.

    method="pos"   -> channels is an (N, 3) RGB array; run POS then bandpass.
    method="green" -> channels is the 1-D green trace (or (N, 3), green taken);
                      detrend then bandpass. The original known-good path.
    hr_band         -> (low_hz, high_hz) analysis band. Defaults to 42-180 BPM.
    """
    t = np.asarray(t, float)
    channels = np.asarray(channels, float)
    lo, hi = hr_band
    if not 0 < lo < hi:
        raise ValueError("hr_band must be an increasing positive frequency range")
    if len(t) < 20 or t[-1] - t[0] < min_seconds:
        return None
    fs = measured_fs(t)
    if fs < 5.0:   # below this the HR band is mostly gone
        return None
    tu = uniform_grid(t, fs)

    if method == "pos":
        if channels.ndim != 2 or channels.shape[1] != 3:
            raise ValueError("POS needs an (N, 3) RGB array")
        rgb = np.column_stack([np.interp(tu, t, channels[:, i]) for i in range(3)])
        raw = pos_signal(rgb, fs)
        filtered = bandpass(raw, fs, lo=lo, hi=hi)
    elif method == "green":
        g = channels if channels.ndim == 1 else channels[:, 1]
        raw = np.interp(tu, t, g)
        filtered = bandpass(detrend(raw, fs), fs, lo=lo, hi=hi)
    else:
        raise ValueError(f"unknown method {method!r}")

    freqs, power = spectrum(filtered, fs)
    f_peak, snr = pick_peak(freqs, power, lo=lo, hi=hi)
    if f_peak is None:
        return None
    dominance = peak_dominance(freqs, power, f_peak, lo=lo, hi=hi)
    return {"bpm": 60.0 * f_peak, "snr": snr, "dominance": dominance,
            "fs": fs, "method": method,
            "hr_band": (lo, hi),
            "t": tu, "raw": raw, "filtered": filtered,
            "freqs": freqs, "power": power}
