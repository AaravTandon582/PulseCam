# Pulse Cam

Contactless heart-rate monitoring from a standard webcam.

No sensors. No wearables. No training data. Just signal processing.

Point a laptop camera at your face, hold still for 30 seconds, and it reads your pulse. It does that by measuring color changes in your skin that are way too faint to see.

---

## How it works

Every heartbeat pushes a pulse of blood into the capillaries in your face. That slightly changes how much light your skin absorbs. The effect is invisible to the eye, but a camera picks it up frame after frame as a tiny periodic ripple in the average color of a patch of skin. This is **remote photoplethysmography (rPPG)**.

The pipeline:

```
Webcam -> face detection + skin ROI -> per-frame RGB means ->
timestamp-aware resampling -> POS pulse extraction ->
detrend + bandpass + FFT -> quality gate -> stabilized BPM
```

Every arrow there is a real engineering decision. The interesting ones are below.

**No machine learning is involved.** This is deterministic DSP end to end, and that's deliberate. The signal is physical and well-characterized, so a filter chain beats a model with no training data behind it.

---

## Quick start

```bash
git clone <repo>
cd pulse-cam
pip install -r requirements.txt
python main.py
```

macOS will ask for camera permission the first time. If it doesn't, grant it manually under System Settings > Privacy & Security > Camera and restart the terminal.

### Flags

| Flag | What it does |
|---|---|
| `--camera N` | Pick a camera index if you have more than one |
| `--roi forehead\|cheeks\|both` | Choose which skin region to sample |
| `--video PATH` | Replay a recorded clip at real timing instead of using the webcam |
| `--dump` | Save timestamps, raw trace, filtered trace, and spectrum to `.npz` + `.png` |

`--video` is the one that saves you time. Record once, tune forever. You get a fixed input to tune the filters against, instead of chasing a signal that changes every run.

---

## Signal extraction: POS, not green-channel

The naive version of rPPG takes the mean of the green channel, since hemoglobin absorbs green most strongly. It works in ideal conditions and falls apart the moment you move or the lighting flickers.

This uses **POS (Plane-Orthogonal-to-Skin, Wang et al.)** instead. POS uses all three RGB channels and projects onto a plane chosen so that motion and illumination artifacts cancel out, since those affect all channels roughly together. The pulse signal survives the projection.

The difference is measurable. In adversarial synthetic tests, where a motion artifact was injected alongside a known pulse, green-only locked onto the artifact and reported it confidently as a heart rate. POS recovered the actual rate.

---

## Three problems that took most of the build

### 1. Your webcam is not running at 30fps

It says 30fps. Under CPU load from live face tracking, it isn't. It might be running at 15, and it might drift within a single session.

This matters more than it sounds like it should. Every frequency-domain calculation assumes you know the sample rate. Assume 30 when you're really getting 15, and the FFT hands back a confident, precisely wrong BPM with no warning at all. That's the worst failure mode there is.

The fix: every frame gets a `time.monotonic()` timestamp at capture. The true sample rate is computed from those timestamps, and the buffer is resampled onto a uniform grid before anything touches the FFT.

### 2. The DSP has to be testable without a camera

`dsp.py` is pure. Numpy and scipy only, no OpenCV, no I/O, no state. It takes an array in and gives numbers back.

That means `test_dsp.py` can generate synthetic pulse signals at known rates across 48 to 150 BPM, including deliberately jittered frame timing, and verify the whole chain recovers the right answer. Those tests passed before a single live reading was trusted.

Which paid for itself immediately. Once the math is verified, any wrong live reading is a *signal quality* problem (bad ROI, bad lighting, motion) and not a DSP bug. That cuts the debugging surface in half and tells you which half.

### 3. Never print a confident wrong number

`quality.py` scores the SNR of the spectral peak, ROI brightness, and landmark motion. Below threshold, the display shows "hold still" or "need more light" instead of a BPM. During the first ten seconds it shows a collecting-data counter.

An honest "I don't know yet" is worth more than a plausible number that's wrong.

---

## Demo hardening

Three bugs found while stress-testing the demo flow, all fixed:

- **Frozen readout.** "CHECK COMPLETE" held stale data on screen indefinitely, so a number from two minutes ago looked live. Now there's a 12-second idle timeout with dimming.
- **Face dropout mid-check.** Walk out of frame for ten seconds and the check would still "complete," having interpolated straight across the gap. Now it tracks seconds of *valid detected face*, not wall-clock time.
- **Camera failure.** Unplug the camera and you got a raw Python traceback. Now it's a clean on-screen error card.

---

## Validating it yourself

Don't take the number on faith. Count your own pulse at your wrist or neck for 30 seconds, double it, and compare against what the app reports over the same window.

In repeated live sessions the stabilized reading held within about **±3 BPM** of a manual count. That's in the range of normal beat-to-beat variation in a resting heart rate, not just measurement noise.

---

## Layout

| File | Role |
|---|---|
| `main.py` | Entry point, wires the stages together, CLI |
| `capture.py` | Webcam or recorded-clip frames, plus a monotonic timestamp per frame |
| `face_region.py` | MediaPipe FaceMesh landmarks to forehead + cheek ROI mask |
| `signal_extract.py` | Spatial RGB means over the ROI, into a ring buffer |
| `dsp.py` | Pure DSP: resample, detrend, bandpass, FFT, peak, BPM |
| `quality.py` | SNR, brightness, and motion checks, into gating messages |
| `overlay.py` | Draws ROI, BPM, and status onto the frame |
| `liveplot.py` | Raw trace, filtered trace, and spectrum as OpenCV panels |
| `debug_dump.py` | `--dump` output for offline inspection |
| `test_dsp.py` | Synthetic-signal test suite |

Live plots are rendered directly onto an OpenCV canvas rather than through matplotlib, because matplotlib drops frames badly at 30fps on CPU. It's used only for offline debug dumps.

If MediaPipe fails to load, `face_region.py` falls back to an OpenCV Haar cascade with a geometric forehead box. Degraded, but a dependency hiccup doesn't take down the demo.

---

## Where this is useful

- Telehealth, where the provider can't physically touch the patient
- Patients you don't want to attach hardware to, like infants or burn patients
- Any situation where there's a camera available and no wearable

---

## Limitations

This is a wellness and technology demonstration. **It is not a medical device.**

- Accuracy depends on steady lighting, limited head motion, face size in frame, and webcam quality
- It doesn't diagnose anything, detect arrhythmias, or measure blood oxygen
- POS made it meaningfully more robust to motion and lighting, but not universally robust. Full real-world robustness is still an open research problem.

## Future work

- **Respiration rate.** The cheapest real addition, since breathing shows up as a slow oscillation in the same trace. A second bandpass around 0.15 to 0.4 Hz reuses almost the whole existing chain.
- **HRV.** Needs precise beat-to-beat timing, and webcam rPPG probably isn't clean enough to claim it credibly. Listed here rather than built, for that reason.
- **SpO2.** Requires calibrated multi-wavelength analysis. Not honestly doable on a laptop webcam.

---

Built in 7.5 hours on an M1 MacBook Air, CPU only.
