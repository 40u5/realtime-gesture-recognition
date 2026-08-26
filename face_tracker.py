"""Face tracking with MediaPipe Face Mesh: nose position and mouth state."""

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

# Averaged cluster: expressions and eye movement wobble the mesh slightly,
# and the nose region is its most stable part.
NOSE = (1, 2, 4)

# (upper inner lip, lower inner lip)
MOUTH = (13, 14)


@dataclass
class FaceObservation:
    nose_x: float  # normalized frame coords (0..1) in the mirrored frame
    nose_y: float
    mouth_ratio: float  # lip gap / philtrum: ~0.1 closed, >1 wide open
    markers: dict  # name -> (px, py), for drawing


class FaceTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_bgr) -> FaceObservation | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark
        pt = lambda i: np.array([lm[i].x, lm[i].y])

        nose = np.mean([pt(i) for i in NOSE], axis=0)

        lip_top, lip_bottom = pt(MOUTH[0]), pt(MOUTH[1])
        # Reference the lip gap against nose-to-upper-lip, not mouth width:
        # both are vertical spans, so pitching the head down foreshortens them
        # equally and the ratio holds. Mouth width does not compress with
        # pitch, which made opens undetectable while looking down.
        philtrum = float(np.linalg.norm(lip_top - nose)) + 1e-9
        mouth_ratio = float(np.linalg.norm(lip_bottom - lip_top)) / philtrum

        h, w = frame_bgr.shape[:2]
        px = lambda p: (int(p[0] * w), int(p[1] * h))

        markers = {
            "nose": px(nose),
            "mouth": px((lip_top + lip_bottom) / 2.0),
        }

        return FaceObservation(
            nose_x=float(nose[0]),
            nose_y=float(nose[1]),
            mouth_ratio=mouth_ratio,
            markers=markers,
        )

    def close(self):
        self.face_mesh.close()
