"""Presentation overlay for the camera view and existing quality-gate state."""
import cv2
import numpy as np

GREEN = (80, 220, 80)
YELLOW = (50, 200, 255)
RED = (70, 80, 240)
WHITE = (240, 240, 240)
MUTED = (175, 175, 175)
DARK = (18, 18, 18)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _signal_style(status, status_ok):
    """Map existing quality messages to a short, demo-readable state label."""
    if status_ok:
        return "SIGNAL GOOD", GREEN
    lowered = status.lower()
    if "collecting" in lowered:
        return "COLLECTING DATA", YELLOW
    if "light" in lowered or "bright" in lowered:
        return "CHECK LIGHTING", YELLOW
    if "face" in lowered:
        return "FACE NOT FOUND", RED
    if "motion" in lowered or "still" in lowered:
        return "HOLD STILL", YELLOW
    if "ambiguous" in lowered or "weak" in lowered:
        return "SIGNAL WEAK", YELLOW
    if "consensus" in lowered or "jump" in lowered or "stabil" in lowered:
        return "STABILIZING", YELLOW
    if "press r" in lowered:
        return "PRESS R FOR NEW CHECK", YELLOW
    return "SIGNAL CHECK", YELLOW


def draw(frame, polys, fps, bpm_text, status, status_ok, check_progress=0.0,
         check_duration=30.0, check_complete=False, stale=False):
    """Add a legible BPM card and quality indicator without hiding the ROI."""
    for p in polys:
        cv2.polylines(frame, [p], True, GREEN, 2)
    h, w = frame.shape[:2]
    label, color = _signal_style(status, status_ok)

    # Main card: readable from across a table during the demo.
    card_right = min(w - 10, 315)
    fps_text = f"{fps:4.1f} FPS"
    (tw, _), _ = cv2.getTextSize(fps_text, FONT, 0.5, 1)

    # Blend all HUD backgrounds at once. Repeated full-frame copies here can
    # reduce capture FPS enough to harm the rPPG signal on a CPU-only laptop.
    layer = frame.copy()
    cv2.rectangle(layer, (10, 10), (card_right, 152), DARK, -1)
    cv2.rectangle(layer, (w - tw - 30, 10), (w - 10, 38), DARK, -1)
    cv2.rectangle(layer, (0, h - 28), (w, h), DARK, -1)
    cv2.addWeighted(layer, 0.72, frame, 0.28, 0, frame)

    title = "CHECK COMPLETE" if check_complete else "30-SECOND PULSE CHECK"
    cv2.putText(frame, title, (24, 34), FONT, 0.52,
                GREEN if check_complete and not stale else MUTED, 1, cv2.LINE_AA)
    bpm_color = MUTED if stale else (GREEN if status_ok else WHITE)
    cv2.putText(frame, bpm_text, (22, 82), FONT, 1.55,
                bpm_color, 3, cv2.LINE_AA)
    cv2.rectangle(frame, (22, 96), (card_right - 14, 116), color, -1)
    cv2.putText(frame, label, (29, 111), FONT, 0.47, DARK, 1, cv2.LINE_AA)
    elapsed = check_duration if check_complete else check_progress * check_duration
    cv2.putText(frame, f"MEASUREMENT {elapsed:04.1f} / {check_duration:.0f} s",
                (24, 137), FONT, 0.42, MUTED, 1, cv2.LINE_AA)
    bar_left, bar_right = 24, card_right - 16
    cv2.rectangle(frame, (bar_left, 142), (bar_right, 147), (65, 65, 65), -1)
    fill = int(bar_left + (bar_right - bar_left) * min(1.0, check_progress))
    cv2.rectangle(frame, (bar_left, 142), (fill, 147), GREEN if check_complete else color, -1)

    # Small persistent technical context, useful without competing with BPM.
    cv2.putText(frame, fps_text, (w - tw - 20, 30), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    footer = "CONTACTLESS HEART RATE  |  FACE rPPG"
    cv2.putText(frame, footer, (14, h - 9), FONT, 0.43, MUTED, 1, cv2.LINE_AA)
    return frame


def show_fatal_error(message, window="pulse-cam"):
    """Clean full-window error card instead of a raw traceback; any key exits."""
    img = np.full((240, 640, 3), DARK[0], np.uint8)
    cv2.putText(img, "CAMERA ERROR", (24, 56), FONT, 0.85, RED, 2, cv2.LINE_AA)
    cv2.putText(img, message, (24, 116), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, "check camera permissions, or try --camera 1 / --video <clip>",
                (24, 156), FONT, 0.45, MUTED, 1, cv2.LINE_AA)
    cv2.putText(img, "press any key to exit", (24, 196), FONT, 0.45, MUTED, 1,
                cv2.LINE_AA)
    cv2.imshow(window, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
