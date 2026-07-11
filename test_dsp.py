"""Synthetic-signal test for dsp.py — must pass before any live debugging.

Builds a fake pulse with everything the webcam will throw at us: timestamp
jitter, slow lighting drift, an auto-exposure ramp, and sensor noise. If this
passes, a bad live BPM is a signal-quality problem upstream, not the math.
"""
import numpy as np

import dsp


def synth(bpm, seconds=15.0, fs_nominal=30.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * fs_nominal)
    t = np.sort(np.arange(n) / fs_nominal + rng.normal(0, 0.004, n))
    f = bpm / 60.0
    v = (np.sin(2 * np.pi * f * t)              # the pulse
         + 5.0 * np.sin(2 * np.pi * 0.05 * t)   # slow lighting drift
         + 0.08 * t * fs_nominal / 30.0         # auto-exposure ramp
         + rng.normal(0, 0.3, n))               # sensor noise
    return t, v


def check(bpm, **kwargs):
    t, v = synth(bpm, **kwargs)
    r = dsp.estimate(t, v)
    if r is None:
        print(f"FAIL  target {bpm:6.1f}  no estimate returned")
        return False
    err = abs(r["bpm"] - bpm)
    ok = err <= 1.0
    print(f"{'PASS' if ok else 'FAIL'}  target {bpm:6.1f}  got {r['bpm']:6.1f}"
          f"  (err {err:4.2f}, snr {r['snr']:5.1f}, fs {r['fs']:4.1f})")
    return ok


if __name__ == "__main__":
    results = [check(b) for b in (48, 72, 110, 150)]
    results.append(check(72, fs_nominal=24.0, seed=1))   # slow camera
    raise SystemExit(0 if all(results) else 1)
