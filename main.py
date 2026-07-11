"""pulse-cam: contactless heart-rate monitor (rPPG).

Webcam (or replayed clip) -> face ROI -> RGB means -> DSP (POS or green) -> BPM,
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
import session_log
import signal_extract

DSP_INTERVAL = 0.5   # seconds between BPM recomputes (~2x/sec)
DRIFT_CONFIRMATIONS = 3
DRIFT_AGREEMENT_BPM = 6.0


def parse_args():
    ap = argparse.ArgumentParser(description="contactless heart-rate monitor (rPPG)")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--video", help="replay a recorded clip instead of the webcam")
    ap.add_argument("--record", help="save the live feed to this .mp4 while running")
    ap.add_argument("--roi", choices=["forehead", "cheeks", "both"], default="both")
    ap.add_argument("--method", choices=["pos", "green"], default="pos",
                    help="pulse extraction: POS (robust, default) or green mean")
    ap.add_argument("--window", type=float, default=15.0,
                    help="FFT window length in seconds (default: 15)")
    ap.add_argument("--max-bpm", type=float, default=180.0,
                    help="upper analysis limit in BPM (default: 180)")
    ap.add_argument("--smooth", type=int, default=9,
                    help="median smoothing estimates to retain (default: 9)")
    ap.add_argument("--max-jump", type=float, default=18.0,
                    help="reject a reading this many BPM from the current value "
                         "(default: 18)")
    ap.add_argument("--check-duration", type=float, default=30.0,
                    help="seconds to collect before a successful check completes "
                         "(default: 30)")
    ap.add_argument("--session-dir", default="sessions",
                    help="directory for completed check metadata (default: sessions)")
    ap.add_argument("--no-save-session", action="store_true",
                    help="do not write a JSON record when a check completes")
    ap.add_argument("--dump", action="store_true",
                    help="save signals to dumps/ (.npz + .png) on exit")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.window <= 0:
        raise ValueError("--window must be positive")
    if args.max_bpm <= dsp.HR_BAND[0] * 60:
        raise ValueError(f"--max-bpm must exceed {dsp.HR_BAND[0] * 60:.0f}")
    if args.smooth < 1:
        raise ValueError("--smooth must be at least 1")
    if args.max_jump < 0:
        raise ValueError("--max-jump cannot be negative")
    if args.check_duration < args.window:
        raise ValueError("--check-duration must be at least --window")
    src = capture.FrameSource(camera=args.camera, video=args.video)
    finder = face_region.make_region_finder(args.roi)
    recorder = capture.ClipRecorder(args.record) if args.record else None
    def new_check():
        return (signal_extract.SignalBuffer(seconds=args.check_duration + 5.0),
                deque(maxlen=args.smooth), deque(maxlen=DRIFT_CONFIRMATIONS))

    buf, bpm_history, pending_jump = new_check()

    fps = 0.0
    prev_t = None
    last_dsp = 0.0
    result = None
    check_complete = False
    completed_bpm = None
    session_saved = False
    shown_bpm = None
    session_bpms = []

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
            tt, rr, gg, bb = buf.arrays_rgb()
            keep = tt >= tt[-1] - args.window
            if args.method == "pos":
                channels = np.column_stack([rr, gg, bb])[keep]
            else:
                channels = gg[keep]
            r = dsp.estimate(tt[keep], channels, method=args.method,
                             hr_band=(dsp.HR_BAND[0], args.max_bpm / 60.0))
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
        if ok and result is not None and not check_complete:
            candidate = result["bpm"]
            if shown_bpm is None or abs(candidate - shown_bpm) <= args.max_jump:
                bpm_history.append(candidate)
                pending_jump.clear()
                shown_bpm = float(np.median(bpm_history))
                session_bpms.append(shown_bpm)
            else:
                # Keep a one-off spectral jump from replacing a plausible reading.
                pending_jump.append(candidate)
                agreeing = (len(pending_jump) == DRIFT_CONFIRMATIONS
                            and max(pending_jump) - min(pending_jump)
                            <= DRIFT_AGREEMENT_BPM)
                if agreeing:
                    bpm_history.clear()
                    bpm_history.extend(pending_jump)
                    pending_jump.clear()
                    shown_bpm = float(np.median(bpm_history))
                    session_bpms.append(shown_bpm)
                else:
                    ok, status = False, "stabilizing reading - hold still"
        measurement_bpm = (session_log.robust_bpm(session_bpms)
                           if session_bpms else shown_bpm)
        if (not check_complete and measurement_bpm is not None and span >= args.check_duration
                and ok and result is not None):
            check_complete = True
            completed_bpm = measurement_bpm
            status, status_ok = "check complete", True
            if not args.no_save_session and not session_saved:
                path = session_log.save(args.session_dir, completed_bpm,
                                        args.check_duration, result, args.method)
                print(f"pulse check saved: {path}")
                session_saved = True
        if check_complete:
            measurement_bpm = completed_bpm
            status, ok = "check complete", True
        bpm_text = (f"{measurement_bpm:.0f} BPM"
                    if measurement_bpm is not None else "-- BPM")

        display = overlay.draw(frame, roi.polys if roi else [], fps,
                               bpm_text, status, ok,
                               check_progress=min(1.0, span / args.check_duration),
                               check_duration=args.check_duration,
                               check_complete=check_complete)
        stack = np.vstack([display] + liveplot.panels(buf, result, display.shape[1]))
        cv2.imshow("pulse-cam", stack)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("r"):
            buf, bpm_history, pending_jump = new_check()
            result = None
            last_dsp = t
            check_complete = False
            completed_bpm = None
            session_saved = False
            shown_bpm = None
            session_bpms = []

    src.release()
    if recorder:
        recorder.close()
    cv2.destroyAllWindows()
    if args.dump and result is not None:
        import debug_dump
        print("dumped:", debug_dump.dump(result))


if __name__ == "__main__":
    main()
