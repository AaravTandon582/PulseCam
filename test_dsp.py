"""Synthetic-signal test for dsp.py — must pass before any live debugging.

Two parts:
  1. Single-channel (green) path: a pulse buried under jitter, slow lighting
     drift, an auto-exposure ramp and noise still recovers the right BPM.
  2. POS vs green under an adversarial common-mode artifact: a motion/lighting
     oscillation shared across R, G, B — stronger than the pulse and inside the
     HR band. POS cancels it and finds the pulse; the plain green mean is pulled
     onto the artifact. This is the whole point of POS.
"""
import numpy as np

import dsp


# ---------- part 1: single-channel green path ----------

def synth_green(bpm, seconds=15.0, fs_nominal=30.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * fs_nominal)
    t = np.sort(np.arange(n) / fs_nominal + rng.normal(0, 0.004, n))
    f = bpm / 60.0
    v = (np.sin(2 * np.pi * f * t)              # the pulse
         + 5.0 * np.sin(2 * np.pi * 0.05 * t)   # slow lighting drift
         + 0.08 * t * fs_nominal / 30.0         # auto-exposure ramp
         + rng.normal(0, 0.3, n))               # sensor noise
    return t, v


def check_green(bpm, **kwargs):
    t, v = synth_green(bpm, **kwargs)
    r = dsp.estimate(t, v, method="green")
    return _report("green", bpm, r)


# ---------- part 2: 3-channel POS-vs-green ----------

def synth_rgb(bpm, motion_bpm=102.0, seconds=20.0, fs_nominal=30.0, seed=3):
    """RGB skin trace: a channel-specific pulse plus a strong common-mode
    motion/lighting oscillation shared identically across all three channels."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs_nominal)
    t = np.sort(np.arange(n) / fs_nominal + rng.normal(0, 0.004, n))
    f_pulse = bpm / 60.0
    f_motion = motion_bpm / 60.0

    dc = np.array([0.60, 0.50, 0.40])           # skin tone R, G, B
    pulse_gain = np.array([0.004, 0.010, 0.007])   # channel-specific pulse AC
    pulse = np.sin(2 * np.pi * f_pulse * t)
    motion = np.sin(2 * np.pi * f_motion * t)   # common-mode, 3x the pulse
    slow = np.sin(2 * np.pi * 0.05 * t)         # common-mode slow drift

    chans = []
    for c in range(3):
        signal = dc[c] * (1.0
                          + pulse_gain[c] * pulse
                          + 0.030 * motion
                          + 0.050 * slow
                          + rng.normal(0, 0.002, n))
        chans.append(signal)
    return t, np.column_stack(chans)            # (N, 3)


def check_pos_vs_green(bpm, motion_bpm=102.0, **kwargs):
    t, rgb = synth_rgb(bpm, motion_bpm=motion_bpm, **kwargs)
    r_pos = dsp.estimate(t, rgb, method="pos")
    r_green = dsp.estimate(t, rgb, method="green")   # green col of the same RGB
    pos_ok = _report("POS  ", bpm, r_pos)
    # green is expected to be pulled toward the motion artifact, not the pulse
    g_bpm = r_green["bpm"] if r_green else float("nan")
    green_fooled = r_green is not None and abs(g_bpm - bpm) > 5.0
    print(f"      green mean on same RGB -> {g_bpm:6.1f} BPM  "
          f"({'fooled by motion artifact, as expected' if green_fooled else 'NOT fooled'})")
    return pos_ok and green_fooled


def _report(tag, bpm, r):
    if r is None:
        print(f"FAIL  [{tag}] target {bpm:6.1f}  no estimate returned")
        return False
    err = abs(r["bpm"] - bpm)
    ok = err <= 1.5
    print(f"{'PASS' if ok else 'FAIL'}  [{tag}] target {bpm:6.1f}  "
          f"got {r['bpm']:6.1f}  (err {err:4.2f}, snr {r['snr']:5.1f}, "
          f"fs {r['fs']:4.1f})")
    return ok


if __name__ == "__main__":
    results = []
    print("-- green path (clean-ish single channel) --")
    results += [check_green(b) for b in (48, 72, 110, 150)]
    results.append(check_green(72, fs_nominal=24.0, seed=1))   # slow camera

    print("\n-- POS vs green (adversarial common-mode motion @ 102 BPM) --")
    results.append(check_pos_vs_green(72))
    results.append(check_pos_vs_green(90, motion_bpm=132.0, seed=5))

    print(f"\n{sum(results)}/{len(results)} passed")
    raise SystemExit(0 if all(results) else 1)
