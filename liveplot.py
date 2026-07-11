"""Instrumentation panels rendered with OpenCV (no matplotlib in the live loop):
raw trace, filtered trace, and FFT spectrum, stacked under the camera view."""
import cv2
import numpy as np

from dsp import HR_BAND

PANEL_H = 110
BG = 22
GRID = (60, 60, 60)
RAW_C = (200, 180, 80)
FILT_C = (90, 220, 120)
SPEC_C = (80, 160, 250)
PEAK_C = (60, 60, 230)
TEXT = (200, 200, 200)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _canvas(w, title):
    img = np.full((PANEL_H, w, 3), BG, np.uint8)
    cv2.line(img, (0, 0), (w, 0), GRID, 1)
    cv2.putText(img, title, (8, 16), FONT, 0.45, TEXT, 1, cv2.LINE_AA)
    return img


def trace_panel(v, w, title, color):
    img = _canvas(w, title)
    v = np.asarray(v, float)
    if len(v) >= 2:
        lo, hi = v.min(), v.max()
        span = (hi - lo) or 1.0
        ys = (PANEL_H - 10) - (v - lo) / span * (PANEL_H - 34)
        xs = np.linspace(6, w - 6, len(v))
        pts = np.column_stack([xs, ys]).astype(np.int32)
        cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "waiting for data...", (w // 2 - 70, PANEL_H // 2),
                    FONT, 0.5, TEXT, 1, cv2.LINE_AA)
    return img


def spectrum_panel(freqs, power, bpm, w, hr_band=HR_BAND):
    img = _canvas(w, f"spectrum ({hr_band[0] * 60:.0f}-{hr_band[1] * 60:.0f} BPM)")
    if freqs is None:
        cv2.putText(img, "waiting for data...", (w // 2 - 70, PANEL_H // 2),
                    FONT, 0.5, TEXT, 1, cv2.LINE_AA)
        return img
    band = (freqs >= hr_band[0]) & (freqs <= hr_band[1])
    f, p = freqs[band], power[band]
    if len(f) >= 2 and p.max() > 0:
        xs = 6 + (f - f[0]) / (f[-1] - f[0]) * (w - 12)
        ys = (PANEL_H - 10) - p / p.max() * (PANEL_H - 34)
        pts = np.column_stack([xs, ys]).astype(np.int32)
        cv2.polylines(img, [pts], False, SPEC_C, 1, cv2.LINE_AA)
        if bpm is not None:
            x = int(6 + (bpm / 60.0 - f[0]) / (f[-1] - f[0]) * (w - 12))
            cv2.line(img, (x, 22), (x, PANEL_H - 6), PEAK_C, 1)
            cv2.putText(img, f"{bpm:.0f}", (min(x + 5, w - 45), 34),
                        FONT, 0.5, PEAK_C, 1, cv2.LINE_AA)
    return img


def panels(buf, result, w):
    """The three instrumentation panels for the current state."""
    _, raw_live = buf.arrays() if buf.span() > 0 else (None, [])
    out = [trace_panel(raw_live, w, "raw green trace", RAW_C)]
    if result is not None:
        hr_band = result.get("hr_band", HR_BAND)
        out.append(trace_panel(result["filtered"], w,
                               f"filtered {hr_band[0]}-{hr_band[1]} Hz  "
                               f"(fs={result['fs']:.1f})", FILT_C))
        out.append(spectrum_panel(result["freqs"], result["power"],
                                  result["bpm"], w, hr_band))
    else:
        out.append(trace_panel([], w, "filtered", FILT_C))
        out.append(spectrum_panel(None, None, None, w))
    return out
