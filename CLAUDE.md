# pulse-cam — contactless heart-rate monitor (rPPG)

Webcam → face ROI → green-channel mean → bandpass 0.7–4 Hz → FFT peak → BPM.
Hackathon project (7.5h). Win condition: live demo reading a real pulse off a face.
M1 MacBook Air, CPU only, Python (≤3.12 for MediaPipe), ~30fps real time.

## Iron rules
1. **Freeze before improving.** Before adding ANY improvement to a working stage,
   git-commit the working version. A demoable v1 beats a broken v2.
2. **Never assume 30fps.** Every frame is timestamped (`time.monotonic()`); the
   sample rate is computed from timestamps and the signal is resampled to a
   uniform grid before any FFT. The BPM is wrong otherwise — silently.
3. **Instrumentation is not optional.** Raw trace, filtered trace, and FFT
   spectrum must always be viewable live and dumpable via `--dump` (.npz + .png).
   All tuning happens by watching these plots.
4. **Never print a confident wrong BPM.** If signal quality (SNR / motion /
   brightness) is below threshold, show "hold still" or "need more light" and
   freeze the readout. While the buffer is still filling (~first 10s), show
   "collecting data… (Ns / 10s)" — never a blank readout or a premature warning;
   the demo must look alive from second one.
5. `dsp.py` stays pure (numpy/scipy only, no cv2, no I/O) so it's testable
   without a camera. `test_dsp.py` (synthetic 72-BPM sine, jittered timestamps)
   must pass before debugging anything live.
6. **Record once, tune forever.** capture.py can replay a saved clip (`--video`).
   Capture one clean 30s clip early and tune everything against it.

## Build order (each stage verified + committed before the next)
- A: capture + FaceMesh + ROI drawn on screen (forehead/cheeks), fps readout
- B: green-channel spatial mean → (timestamp, value) ring buffer, live raw trace
- C: DSP offline-first — resample → detrend → bandpass 0.7–4 Hz → FFT →
     interpolated peak → BPM; synthetic test passes before live use
- D: live OpenCV plot panels (raw/filtered/spectrum) + quality-gated BPM readout

## Model plan
- **Fable**: planning, architecture decisions, and hard signal-processing
  debugging (wrong BPM, aliasing, filter artifacts).
- **Sonnet**: bulk implementation of the staged files above.

## Demo environment notes
- Steady, diffuse, bright lighting on the face; avoid backlight and flicker.
- Subject sits still, face fills a good part of the frame.
- macOS auto-exposure injects slow drift — detrending handles it, but if the
  raw trace shows sawtooth steps, that's the camera, not the code.
- Tune quality thresholds IN the demo room; budget the last 30 min for it.
- Reference for validating on stage: manual pulse count (or a fingertip
  pulse-ox) is the primary reference. A fitness watch is a rough sanity check
  only — wrist optical HR errs by several BPM itself.

## Tuning knobs (human-in-the-loop, by watching plots)
- ROI: `--roi forehead|cheeks|both`, polygon margins
- Filter: bandpass cutoffs/order, detrend window, FFT window length (default 10s)
- Quality gate: SNR / motion / brightness thresholds
