"""Small, privacy-preserving records for completed pulse-check sessions."""
import json
import os
from statistics import median
import time


def robust_bpm(readings):
    """Session result resistant to a short bad segment at the end of a check."""
    if not readings:
        raise ValueError("at least one quality-approved reading is required")
    return float(median(readings))


def save(out_dir, bpm, duration, result, method):
    """Write metadata only; no frames, face landmarks, or raw signal are saved."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, time.strftime("pulse-check-%Y%m%d-%H%M%S.json"))
    record = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "bpm": round(float(bpm), 1),
        "measurement_seconds": round(float(duration), 1),
        "method": method,
        "sample_rate_hz": round(float(result["fs"]), 2),
        "snr": round(float(result["snr"]), 2),
        "peak_dominance": round(float(result["dominance"]), 2),
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
        file.write("\n")
    return path
