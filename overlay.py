"""Draws the ROI, BPM readout, status message, and measured fps on the frame."""
import cv2

GREEN = (80, 220, 80)
YELLOW = (50, 200, 255)
WHITE = (240, 240, 240)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw(frame, polys, fps, bpm_text, status, status_ok):
    for p in polys:
        cv2.polylines(frame, [p], True, GREEN, 2)
    cv2.putText(frame, f"{fps:4.1f} fps", (frame.shape[1] - 110, 26),
                FONT, 0.6, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, bpm_text, (16, 52),
                FONT, 1.4, GREEN if status_ok else WHITE, 3, cv2.LINE_AA)
    if not status_ok:
        cv2.putText(frame, status, (16, 88), FONT, 0.8, YELLOW, 2, cv2.LINE_AA)
    return frame
