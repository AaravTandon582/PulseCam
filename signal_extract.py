"""Spatial mean of the ROI -> ring buffer of (timestamp, r, g, b).

Buffers all three channels so the DSP can run POS (which needs RGB); the
green-only trace is still exposed via arrays() for the green fallback path and
the live raw panel.
"""
from collections import deque

import cv2
import numpy as np


class SignalBuffer:
    def __init__(self, seconds=15.0):
        self.seconds = seconds
        self.t = deque()
        self.r = deque()
        self.g = deque()
        self.b = deque()
        self.brightness = 0.0   # mean ROI brightness of the latest frame, 0-255

    def add(self, t, frame, mask):
        b, g, r, _ = cv2.mean(frame, mask=mask)   # OpenCV frames are BGR
        self.brightness = (b + g + r) / 3.0
        self.t.append(t)
        self.r.append(r)
        self.g.append(g)
        self.b.append(b)
        while self.t and t - self.t[0] > self.seconds:
            self.t.popleft()
            self.r.popleft()
            self.g.popleft()
            self.b.popleft()

    def span(self):
        return self.t[-1] - self.t[0] if len(self.t) > 1 else 0.0

    def arrays(self):
        """(t, green) — the raw display trace and the green fallback signal."""
        return np.array(self.t), np.array(self.g)

    def arrays_rgb(self):
        """(t, r, g, b) — full RGB for the POS path."""
        return (np.array(self.t), np.array(self.r),
                np.array(self.g), np.array(self.b))
