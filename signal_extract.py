"""Spatial mean of the ROI green channel -> ring buffer of (timestamp, value)."""
from collections import deque

import cv2
import numpy as np


class SignalBuffer:
    def __init__(self, seconds=15.0):
        self.seconds = seconds
        self.t = deque()
        self.v = deque()
        self.brightness = 0.0   # mean ROI brightness of the latest frame, 0-255

    def add(self, t, frame, mask):
        b, g, r, _ = cv2.mean(frame, mask=mask)
        self.brightness = (b + g + r) / 3.0
        self.t.append(t)
        self.v.append(g)   # plethysmographic signal is strongest in green
        while self.t and t - self.t[0] > self.seconds:
            self.t.popleft()
            self.v.popleft()

    def span(self):
        return self.t[-1] - self.t[0] if len(self.t) > 1 else 0.0

    def arrays(self):
        return np.array(self.t), np.array(self.v)
