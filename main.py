"""pulse-cam: contactless heart-rate monitor (rPPG).

Webcam (or replayed clip) -> face ROI -> green-channel mean -> DSP -> BPM,
with live instrumentation panels (raw / filtered / spectrum) stacked below
the camera view. q or ESC quits.
"""
import argparse
from collections import deque

import cv2
import numpy as np

import capture
import dsp
import face_region
import liveplot
import overlay
import quality
import signal_extract

DSP_INTERVAL = 0.5   # seconds between BPM recomputes (~2x/sec)


def parse_args():
    ap = argparse.ArgumentParser(description="contactless heart-rate monitor (rPPG)")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--video", help="replay a recorded clip instead of the webcam")
    ap.add_argument("--record", help="save the live feed to this .mp4 while running")
    ap.add_argument("--roi", choices=["forehead", "cheeks", "both"], default="both")
    ap.add_argument("--window", type=float, default=10.0,
                    help="FFT window length in seconds")
    ap.add_argument("--dump", action="store_true",
                    help="save signals to dumps/ (.npz + .png) on exit")
    return ap.parse_args()


def main():
    args = parse_args()
    src = capture.FrameSource(camera=args.camera, video=args.video)
    finder = face_region.make_region_finder(args.roi)
    recorder = capture.ClipRecorder(args.record) if args.record else None
    buf = signal_extract.SignalBuffer(seconds=args.window + 5.0)

    fps = 0.0
    prev_t = None
    last_dsp = 0.0
    result = None
    bpm_history = deque(maxlen=5)
    shown_bpm = None

    while True:
        t, frame = src.read()
        if frame is None:
            break
        if not args.video:
            frame = cv2.flip(frame, 1)   # mirror the live view
        if recorder:
            recorder.add(t, frame)

        if prev_t is not None and t > prev_t:
            inst = 1.0 / (t - prev_t)
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
        prev_t = t

        roi = finder.find(frame)
        if roi is not None:
            buf.add(t, frame, roi.mask)

        if t - last_dsp >= DSP_INTERVAL and buf.span() >= 5.0:
            last_dsp = t
            tt, vv = buf.arrays()
            keep = tt >= tt[-1] - args.window
            r = dsp.estimate(tt[keep], vv[keep])
            if r is not None:
                result = r

        span = buf.span()
        if span < args.window:
            # startup: the demo must look alive from second one
            ok, status = False, f"collecting data... ({span:.0f}s / {args.window:.0f}s)"
        else:
            ok, status = quality.assess(roi is not None, buf.brightness,
                                        roi.motion if roi else 0.0,
                                        result["snr"] if result else 0.0)
        if ok and result is not None:
            bpm_history.append(result["bpm"])
            shown_bpm = float(np.median(bpm_history))
        bpm_text = f"{shown_bpm:.0f} BPM" if shown_bpm is not None else "-- BPM"

        display = overlay.draw(frame, roi.polys if roi else [], fps,
                               bpm_text, status, ok)
        stack = np.vstack([display] + liveplot.panels(buf, result, display.shape[1]))
        cv2.imshow("pulse-cam", stack)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    src.release()
    if recorder:
        recorder.close()
    cv2.destroyAllWindows()
    if args.dump and result is not None:
        import debug_dump
        print("dumped:", debug_dump.dump(result))


if __name__ == "__main__":
    main()
