"""Face ROI (forehead / cheeks) as a binary mask + polygons for drawing.

Primary: MediaPipe Tasks FaceLandmarker (mp.tasks.vision) — mediapipe >= 0.10
dropped the legacy mp.solutions.face_mesh API. Fallback: OpenCV Haar face box
with a geometric forehead rectangle, so a broken/incompatible mediapipe
install (or a model-download failure) doesn't kill the project (known failure
point #3).
"""
import os
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

# Same 468-point face mesh topology the legacy API used; indices unchanged.
FOREHEAD = [10, 109, 67, 103, 54, 21, 70, 63, 105, 66, 107, 9,
            336, 296, 334, 293, 300, 251, 284, 332, 297, 338]
LEFT_CHEEK = [50, 101, 100, 118, 117, 111, 116, 123, 147, 187, 205, 36]
RIGHT_CHEEK = [280, 330, 329, 347, 346, 340, 345, 352, 376, 411, 425, 266]

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"downloading face landmark model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


@dataclass
class RoiResult:
    mask: np.ndarray          # uint8 binary mask, ROI = 255
    polys: list = field(default_factory=list)   # int32 polygons for drawing
    motion: float = 0.0       # mean landmark displacement / face size, per frame


class FaceMeshRegion:
    def __init__(self, roi_mode="both"):
        import mediapipe as mp
        model_path = _ensure_model()
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
        self.mp = mp
        self.roi_mode = roi_mode
        self._prev_pts = None
        self._frame_idx = 0

    def find(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires a strictly increasing per-frame timestamp (ms);
        # the value only needs to be monotonic, not wall-clock accurate.
        self._frame_idx += 1
        res = self.landmarker.detect_for_video(mp_image, self._frame_idx)
        if not res.face_landmarks:
            self._prev_pts = None
            return None
        lm = res.face_landmarks[0]
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
    except Exception as e:
        print(f"mediapipe unavailable ({e}) — falling back to OpenCV face box")
        return HaarRegion()
