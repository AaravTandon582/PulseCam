"""Face ROI (forehead / cheeks) as a binary mask + polygons for drawing.

Primary: MediaPipe FaceMesh landmarks. Fallback: OpenCV Haar face box with a
geometric forehead rectangle, so a broken mediapipe install doesn't kill the
project (known failure point #3).
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

# FaceMesh landmark index sets; convex hull makes ordering irrelevant.
FOREHEAD = [10, 109, 67, 103, 54, 21, 70, 63, 105, 66, 107, 9,
            336, 296, 334, 293, 300, 251, 284, 332, 297, 338]
LEFT_CHEEK = [50, 101, 100, 118, 117, 111, 116, 123, 147, 187, 205, 36]
RIGHT_CHEEK = [280, 330, 329, 347, 346, 340, 345, 352, 376, 411, 425, 266]


@dataclass
class RoiResult:
    mask: np.ndarray          # uint8 binary mask, ROI = 255
    polys: list = field(default_factory=list)   # int32 polygons for drawing
    motion: float = 0.0       # mean landmark displacement / face size, per frame


class FaceMeshRegion:
    def __init__(self, roi_mode="both"):
        import mediapipe as mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.roi_mode = roi_mode
        self._prev_pts = None

    def find(self, frame):
        h, w = frame.shape[:2]
        res = self.mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            self._prev_pts = None
            return None
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)

        face_size = float(pts[:, 1].max() - pts[:, 1].min())
        motion = 0.0
        if self._prev_pts is not None and face_size > 0:
            motion = float(np.mean(np.linalg.norm(pts - self._prev_pts, axis=1))
                           / face_size)
        self._prev_pts = pts

        polys = []
        if self.roi_mode in ("forehead", "both"):
            polys.append(cv2.convexHull(pts[FOREHEAD].astype(np.int32)))
        if self.roi_mode in ("cheeks", "both"):
            polys.append(cv2.convexHull(pts[LEFT_CHEEK].astype(np.int32)))
            polys.append(cv2.convexHull(pts[RIGHT_CHEEK].astype(np.int32)))

        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, polys, 255)
        return RoiResult(mask=mask, polys=polys, motion=motion)


class HaarRegion:
    """Fallback: face box -> geometric forehead rectangle."""

    def __init__(self):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._prev_center = None

    def find(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
        if len(faces) == 0:
            self._prev_center = None
            return None
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

        center = np.array([x + fw / 2, y + fh / 2])
        motion = 0.0
        if self._prev_center is not None:
            motion = float(np.linalg.norm(center - self._prev_center) / fw)
        self._prev_center = center

        # forehead band: middle 50% of the width, upper part of the box
        rx, ry = int(x + 0.25 * fw), int(y + 0.08 * fh)
        rw, rh = int(0.5 * fw), int(0.22 * fh)
        poly = np.array([[rx, ry], [rx + rw, ry],
                         [rx + rw, ry + rh], [rx, ry + rh]], np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        return RoiResult(mask=mask, polys=[poly], motion=motion)


def make_region_finder(roi_mode="both"):
    try:
        return FaceMeshRegion(roi_mode)
    except ImportError as e:
        print(f"mediapipe unavailable ({e}) — falling back to OpenCV face box")
        return HaarRegion()
