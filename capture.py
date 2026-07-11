"""Frame sources behind one interface: live webcam or recorded clip.

Every frame gets a timestamp. Clips are replayed at their true frame timing so
the DSP sees the same sample rate it would live (iron rule 6: record once,
tune forever).
"""
import time

import cv2


class FrameSource:
    def __init__(self, camera=0, video=None):
        self.is_file = video is not None
        self.cap = cv2.VideoCapture(video if self.is_file else camera)
        if not self.cap.isOpened():
            what = f"clip {video}" if self.is_file else f"camera {camera}"
            raise RuntimeError(f"could not open {what}")
        if self.is_file:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_interval = 1.0 / (fps if fps and fps > 0 else 30.0)
            self._next_t = None

    def read(self):
        """Returns (timestamp, frame); (None, None) at end of stream."""
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        if self.is_file:
            # pace playback and stamp with the clip's ideal frame times
            now = time.monotonic()
            if self._next_t is None:
                self._next_t = now
            if self._next_t > now:
                time.sleep(self._next_t - now)
            t = self._next_t
            self._next_t += self.frame_interval
        else:
            t = time.monotonic()
        return t, frame

    def release(self):
        self.cap.release()


class ClipRecorder:
    """Saves the live feed to a clip. Writer is created lazily once the real
    frame rate has been measured — writing a hardcoded 30fps header would make
    replayed timestamps (and thus the BPM) silently wrong."""

    WARMUP_FRAMES = 30

    def __init__(self, path):
        self.path = path
        self.writer = None
        self._pending = []   # (t, frame) until fps is known

    def add(self, t, frame):
        if self.writer is None:
            self._pending.append((t, frame))
            if len(self._pending) > self.WARMUP_FRAMES:
                ts = [p[0] for p in self._pending]
                fps = (len(ts) - 1) / (ts[-1] - ts[0])
                h, w = frame.shape[:2]
                self.writer = cv2.VideoWriter(
                    self.path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                for _, f in self._pending:
                    self.writer.write(f)
                self._pending = []
        else:
            self.writer.write(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            print(f"recorded clip: {self.path}")
