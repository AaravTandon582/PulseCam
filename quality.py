"""Signal-quality gate: decide whether the BPM readout may update.

Iron rule 4: never print a confident wrong BPM. All thresholds here are
demo-room tuning knobs — adjust them by watching the plots, in the room.
"""

SNR_MIN = 2.5          # spectral peak vs in-band noise floor
MOTION_MAX = 0.010     # mean landmark displacement / face size, per frame
BRIGHT_MIN = 60.0      # mean ROI brightness, 0-255
BRIGHT_MAX = 245.0


def assess(face_found, brightness, motion, snr):
    """Returns (ok, message); ok=True means the readout may update."""
    if not face_found:
        return False, "no face detected"
    if brightness < BRIGHT_MIN:
        return False, "need more light"
    if brightness > BRIGHT_MAX:
        return False, "too bright"
    if motion > MOTION_MAX:
        return False, "hold still"
    if snr < SNR_MIN:
        return False, "weak signal - hold still"
    return True, "ok"
