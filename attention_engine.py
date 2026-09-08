"""
attention_engine.py
--------------------
Core logic for the Smart Attention Monitoring System.

Responsibilities:
- Facial landmark detection (via MediaPipe's FaceLandmarker Tasks API)
- Eye Aspect Ratio (EAR) computation for closure/blink detection
- Mouth Aspect Ratio (MAR) computation for yawn detection
- 3D head-pose estimation (pitch/yaw/roll via OpenCV solvePnP)
- Multi-person tracking
- Per-person auto-calibration of the EAR closed-eye threshold
- Rolling "attention score" calculation per person
- Event logging

This module has no OpenCV window / camera code in it.
"""

from __future__ import annotations

import csv
import math
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError as e:
    raise ImportError(
        "mediapipe is required. Install with: pip install mediapipe"
    ) from e


# --------------------------------------------------------------------------
# MediaPipe model
# --------------------------------------------------------------------------

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"


def ensure_model_downloaded(model_path: Path = MODEL_PATH) -> Path:

    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[setup] Downloading face landmark model to {model_path} ...")

    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as e:
        raise RuntimeError(
            "Failed to download the required MediaPipe face_landmarker model "
            f"from {MODEL_URL}.\n"
            "Check your internet connection, or manually download the file "
            f"and place it at: {model_path}"
        ) from e

    print("[setup] Model downloaded.")

    return model_path


# --------------------------------------------------------------------------
# Landmark indices
# --------------------------------------------------------------------------

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Mouth landmarks used for MAR
MOUTH = [61, 291, 13, 14]

NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_MOUTH = 61
RIGHT_MOUTH = 291


# --------------------------------------------------------------------------
# Head pose landmarks
# --------------------------------------------------------------------------

POSE_LANDMARK_IDS = [
    NOSE_TIP,
    CHIN,
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
    LEFT_MOUTH,
    RIGHT_MOUTH,
]


# --------------------------------------------------------------------------
# Generic 3D face model
# --------------------------------------------------------------------------

MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye outer corner
        (225.0, 170.0, -135.0),      # Right eye outer corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0),     # Right mouth corner
    ],
    dtype=np.float64,
)


# --------------------------------------------------------------------------
# Head pose helpers
# --------------------------------------------------------------------------

def _rotation_matrix_to_euler_angles(
    R: np.ndarray,
) -> tuple[float, float, float]:
    """Convert a 3x3 rotation matrix to (pitch, yaw, roll) in degrees."""

    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0

    return tuple(math.degrees(a) for a in (pitch, yaw, roll))


def estimate_head_pose(
    pts_px: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> tuple[float, float, float]:
    """
    Estimate (pitch, yaw, roll) in degrees for a face's landmarks.
    """

    image_points = np.array(
        [pts_px[i] for i in POSE_LANDMARK_IDS],
        dtype=np.float64,
    )

    focal_length = frame_w
    center = (frame_w / 2.0, frame_h / 2.0)

    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, _translation_vector = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    pitch, yaw, roll = _rotation_matrix_to_euler_angles(
        rotation_matrix
    )

    # Wrap pitch into intuitive range.
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180

    return pitch, yaw, roll


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class EngineConfig:

    # EAR / eye detection
    ear_threshold_default: float = 0.21
    ear_consec_frames: int = 15

    # MAR / yawn detection
    mar_threshold: float = 0.6
    mar_consec_frames: int = 15

    # Head pose
    yaw_threshold_deg: float = 25.0
    pitch_threshold_deg: float = 20.0

    # Attention
    attention_window_sec: float = 10.0

    # Logging
    log_path: str = "logs/attention_log.csv"

    # Multi-person tracking
    max_faces: int = 1
    person_max_match_distance_px: float = 150.0
    person_timeout_sec: float = 2.0

    # Auto-calibration
    calibration_seconds: float = 4.0
    calibration_factor: float = 0.75
    calibration_min_samples: int = 5


# --------------------------------------------------------------------------
# Frame result
# --------------------------------------------------------------------------

@dataclass
class FrameResult:
    """Everything computed for one detected person in one video frame."""

    person_id: int
    timestamp: float

    face_found: bool = True

    # Eye
    ear: Optional[float] = None
    eyes_closed: bool = False

    # Mouth / yawn
    mar: Optional[float] = None
    is_yawning: bool = False
    yawn_count: int = 0

    # Head pose
    looking_away: bool = False
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0

    # Attention
    attention_score: float = 100.0
    state: str = "FOCUSED"

    # Calibration
    calibration_progress: float = 1.0

    # Landmarks
    landmarks: Optional[np.ndarray] = None
    centroid: Optional[np.ndarray] = None


# --------------------------------------------------------------------------
# Per-person state
# --------------------------------------------------------------------------

@dataclass
class _PersonState:

    person_id: int
    first_seen: float
    last_seen: float
    centroid: np.ndarray

    # EAR calibration
    ear_threshold: float = 0.21
    calibrating: bool = True
    calibration_samples: list = field(default_factory=list)

    # Eye state
    closed_frame_streak: int = 0
    was_closed: bool = False

    # Head pose state
    was_looking_away: bool = False

    # Yawn state
    mouth_open_streak: int = 0
    was_yawning: bool = False

    # Counters
    blink_count: int = 0
    yawn_count: int = 0
    drowsy_events: int = 0
    distraction_events: int = 0

    # Attention history
    history: deque = field(default_factory=deque)


# --------------------------------------------------------------------------
# Centroid tracker
# --------------------------------------------------------------------------

class _CentroidTracker:
    """
    Minimal nearest-centroid tracker.

    Assigns a persistent integer ID to each detected face across frames.
    """

    def __init__(
        self,
        max_distance: float,
        timeout_sec: float,
    ):
        self.max_distance = max_distance
        self.timeout_sec = timeout_sec

        self._next_id = 0
        self._last_seen: dict[int, float] = {}
        self._centroids: dict[int, np.ndarray] = {}

    def update(
        self,
        centroids: list[np.ndarray],
        ts: float,
    ) -> list[int]:

        # Expire stale tracks
        stale = [
            pid
            for pid, t in self._last_seen.items()
            if ts - t > self.timeout_sec
        ]

        for pid in stale:
            del self._last_seen[pid]
            del self._centroids[pid]

        assigned: list[Optional[int]] = [None] * len(centroids)
        used_ids: set[int] = set()

        if self._centroids:

            existing_ids = list(self._centroids.keys())

            existing_pts = np.array(
                [self._centroids[pid] for pid in existing_ids]
            )

            # Greedy nearest-neighbor matching
            for i, c in enumerate(centroids):

                dists = np.linalg.norm(
                    existing_pts - c,
                    axis=1,
                )

                order = np.argsort(dists)

                for j in order:

                    pid = existing_ids[j]

                    if (
                        pid not in used_ids
                        and dists[j] < self.max_distance
                    ):
                        assigned[i] = pid
                        used_ids.add(pid)
                        break

        # Create IDs for new faces
        for i in range(len(centroids)):

            if assigned[i] is None:
                assigned[i] = self._next_id
                self._next_id += 1

        # Update tracker
        for i, c in enumerate(centroids):

            pid = assigned[i]

            self._centroids[pid] = c
            self._last_seen[pid] = ts

        return assigned  # type: ignore[return-value]


# --------------------------------------------------------------------------
# Attention Engine
# --------------------------------------------------------------------------

class AttentionEngine:
    """
    Multi-person attention / drowsiness / yawn detection engine.
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        on_person_update: Optional[Callable[[int, dict], None]] = None,
    ):

        self.cfg = config or EngineConfig()
        self.on_person_update = on_person_update

        # Load MediaPipe model
        model_path = ensure_model_downloaded()

        base_options = mp_python.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=max(1, self.cfg.max_faces),
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = mp_vision.FaceLandmarker.create_from_options(
            options
        )

        self._last_ts_ms = -1

        # Person tracker
        self._tracker = _CentroidTracker(
            max_distance=self.cfg.person_max_match_distance_px,
            timeout_sec=self.cfg.person_timeout_sec,
        )

        # Person states
        self._persons: dict[int, _PersonState] = {}

        # Logging
        self._log_file = None
        self._csv_writer = None

        self._init_logging()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _init_logging(self) -> None:

        path = Path(self.cfg.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        new_file = not path.exists()

        self._log_file = open(
            path,
            "a",
            newline="",
        )

        self._csv_writer = csv.writer(self._log_file)

        if new_file:
            self._csv_writer.writerow(
                [
                    "timestamp",
                    "iso_time",
                    "person_id",
                    "event",
                    "ear",
                    "mar",
                    "pitch",
                    "yaw",
                    "roll",
                    "attention_score",
                    "state",
                    "yawn_count",
                ]
            )

    def _log_event(
        self,
        event: str,
        result: FrameResult,
    ) -> None:

        if self._csv_writer is None:
            return

        self._csv_writer.writerow(
            [
                f"{result.timestamp:.3f}",
                time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(result.timestamp),
                ),
                result.person_id,
                event,
                f"{result.ear:.4f}"
                if result.ear is not None
                else "",
                f"{result.mar:.4f}"
                if result.mar is not None
                else "",
                f"{result.pitch:.1f}",
                f"{result.yaw:.1f}",
                f"{result.roll:.1f}",
                f"{result.attention_score:.1f}",
                result.state,
                result.yawn_count,
            ]
        )

        self._log_file.flush()

    def close(self) -> None:

        if self._log_file:
            self._log_file.close()

        if hasattr(self, "landmarker"):
            self.landmarker.close()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _euclidean(
        p1: np.ndarray,
        p2: np.ndarray,
    ) -> float:

        return float(np.linalg.norm(p1 - p2))

    # ------------------------------------------------------------------
    # Eye Aspect Ratio
    # ------------------------------------------------------------------

    def _eye_aspect_ratio(
        self,
        pts: np.ndarray,
        idxs: list[int],
    ) -> float:

        p1, p2, p3, p4, p5, p6 = (
            pts[i] for i in idxs
        )

        vertical_1 = self._euclidean(p2, p6)
        vertical_2 = self._euclidean(p3, p5)

        horizontal = self._euclidean(p1, p4)

        if horizontal == 0:
            return 0.0

        return (
            vertical_1 + vertical_2
        ) / (2.0 * horizontal)

    # ------------------------------------------------------------------
    # Mouth Aspect Ratio
    # ------------------------------------------------------------------

    def _mouth_aspect_ratio(
        self,
        pts: np.ndarray,
        idxs: list[int],
    ) -> float:

        left_corner, right_corner, top_lip, bottom_lip = (
            pts[i] for i in idxs
        )

        vertical = self._euclidean(
            top_lip,
            bottom_lip,
        )

        horizontal = self._euclidean(
            left_corner,
            right_corner,
        )

        if horizontal == 0:
            return 0.0

        return vertical / horizontal

    # ------------------------------------------------------------------
    # Attention scoring
    # ------------------------------------------------------------------

    def _update_attention_score(
        self,
        person: _PersonState,
        timestamp: float,
        penalized: bool,
    ) -> float:

        person.history.append(
            (
                timestamp,
                "bad" if penalized else "good",
            )
        )

        cutoff = (
            timestamp
            - self.cfg.attention_window_sec
        )

        while (
            person.history
            and person.history[0][0] < cutoff
        ):
            person.history.popleft()

        if not person.history:
            return 100.0

        bad = sum(
            1
            for _, s in person.history
            if s == "bad"
        )

        total = len(person.history)

        return round(
            100.0 * (1 - bad / total),
            1,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> list[FrameResult]:

        """
        Run detection on a single BGR frame.

        Returns one FrameResult per tracked person.
        """

        ts = time.time()

        h, w = frame_bgr.shape[:2]

        # BGR -> RGB
        rgb = np.ascontiguousarray(
            frame_bgr[:, :, ::-1]
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        # MediaPipe requires increasing timestamps
        ts_ms = int(ts * 1000)

        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1

        self._last_ts_ms = ts_ms

        # Run face landmark detection
        detection = self.landmarker.detect_for_video(
            mp_image,
            ts_ms,
        )

        if not detection.face_landmarks:
            return []

        # --------------------------------------------------------------
        # Extract landmarks and centroids
        # --------------------------------------------------------------

        all_pts = []
        centroids = []

        for face_landmarks in detection.face_landmarks:

            pts = np.array(
                [
                    [
                        lm.x * w,
                        lm.y * h,
                    ]
                    for lm in face_landmarks
                ],
                dtype=np.float32,
            )

            all_pts.append(pts)

            centroids.append(
                pts[NOSE_TIP]
            )

        # Assign person IDs
        person_ids = self._tracker.update(
            centroids,
            ts,
        )

        results: list[FrameResult] = []

        active_ids = set(person_ids)

        # --------------------------------------------------------------
        # Remove expired persons
        # --------------------------------------------------------------

        for pid in list(self._persons.keys()):

            if (
                pid not in active_ids
                and ts - self._persons[pid].last_seen
                > self.cfg.person_timeout_sec
            ):
                del self._persons[pid]

        # --------------------------------------------------------------
        # Process each person
        # --------------------------------------------------------------

        for pid, pts in zip(
            person_ids,
            all_pts,
        ):

            person = self._persons.get(pid)

            # Create new person
            if person is None:

                person = _PersonState(
                    person_id=pid,
                    first_seen=ts,
                    last_seen=ts,
                    centroid=pts[NOSE_TIP],
                    ear_threshold=self.cfg.ear_threshold_default,
                )

                self._persons[pid] = person

            person.last_seen = ts
            person.centroid = pts[NOSE_TIP]

            # ----------------------------------------------------------
            # Calculate EAR
            # ----------------------------------------------------------

            left_ear = self._eye_aspect_ratio(
                pts,
                LEFT_EYE,
            )

            right_ear = self._eye_aspect_ratio(
                pts,
                RIGHT_EYE,
            )

            ear = (
                left_ear + right_ear
            ) / 2.0

            # ----------------------------------------------------------
            # Calculate MAR
            # ----------------------------------------------------------

            mar = self._mouth_aspect_ratio(
                pts,
                MOUTH,
            )

            # ----------------------------------------------------------
            # Head pose
            # ----------------------------------------------------------

            pitch, yaw, roll = estimate_head_pose(
                pts,
                w,
                h,
            )

            # ----------------------------------------------------------
            # Per-person EAR auto-calibration
            # ----------------------------------------------------------

            if person.calibrating:

                elapsed = (
                    ts - person.first_seen
                )

                person.calibration_samples.append(
                    ear
                )

                if (
                    elapsed
                    >= self.cfg.calibration_seconds
                ):

                    samples = (
                        person.calibration_samples
                    )

                    if (
                        len(samples)
                        >= self.cfg.calibration_min_samples
                    ):

                        baseline_open_ear = float(
                            np.median(samples)
                        )

                        person.ear_threshold = (
                            baseline_open_ear
                            * self.cfg.calibration_factor
                        )

                    person.calibrating = False

                calib_progress = min(
                    1.0,
                    elapsed
                    / self.cfg.calibration_seconds,
                )

                # During calibration we still expose MAR
                result = FrameResult(
                    person_id=pid,
                    timestamp=ts,
                    ear=ear,
                    mar=mar,
                    is_yawning=False,
                    yawn_count=person.yawn_count,
                    pitch=pitch,
                    yaw=yaw,
                    roll=roll,
                    attention_score=100.0,
                    state="CALIBRATING",
                    calibration_progress=calib_progress,
                    landmarks=pts,
                    centroid=person.centroid,
                )

                results.append(result)

                self._publish(result)

                continue

            # ----------------------------------------------------------
            # Eye state
            # ----------------------------------------------------------

            eyes_closed = (
                ear < person.ear_threshold
            )

            # ----------------------------------------------------------
            # Looking-away state
            # ----------------------------------------------------------

            looking_away = (
                abs(yaw)
                > self.cfg.yaw_threshold_deg
                or abs(pitch)
                > self.cfg.pitch_threshold_deg
            )

            # ----------------------------------------------------------
            # Eye closed streak / blink
            # ----------------------------------------------------------

            if eyes_closed:

                person.closed_frame_streak += 1

            else:

                if (
                    person.was_closed
                    and person.closed_frame_streak
                    < self.cfg.ear_consec_frames
                ):
                    person.blink_count += 1

                person.closed_frame_streak = 0

            # Drowsiness
            is_drowsy = (
                person.closed_frame_streak
                >= self.cfg.ear_consec_frames
            )

            # ----------------------------------------------------------
            # Yawn detection
            # ----------------------------------------------------------

            if mar > self.cfg.mar_threshold:

                person.mouth_open_streak += 1

            else:

                person.mouth_open_streak = 0

            is_yawning = (
                person.mouth_open_streak
                >= self.cfg.mar_consec_frames
            )

            # Count only when entering a new yawn
            if (
                is_yawning
                and not person.was_yawning
            ):

                person.yawn_count += 1

            # Update previous yawn state
            person.was_yawning = is_yawning

            # ----------------------------------------------------------
            # Determine attention state
            # ----------------------------------------------------------

            if is_drowsy:

                state = "DROWSY"

            elif looking_away:

                state = "DISTRACTED"

            else:

                state = "FOCUSED"

            # ----------------------------------------------------------
            # Attention score
            # ----------------------------------------------------------

            penalized = (
                is_drowsy
                or looking_away
            )

            score = self._update_attention_score(
                person,
                ts,
                penalized,
            )

            # ----------------------------------------------------------
            # Create frame result
            # ----------------------------------------------------------

            result = FrameResult(
                person_id=pid,
                timestamp=ts,

                ear=ear,
                eyes_closed=eyes_closed,

                mar=mar,
                is_yawning=is_yawning,
                yawn_count=person.yawn_count,

                looking_away=looking_away,

                pitch=pitch,
                yaw=yaw,
                roll=roll,

                attention_score=score,
                state=state,

                calibration_progress=1.0,

                landmarks=pts,
                centroid=person.centroid,
            )

            # ----------------------------------------------------------
            # Event logging
            # ----------------------------------------------------------

            if (
                is_drowsy
                and person.closed_frame_streak
                == self.cfg.ear_consec_frames
            ):

                person.drowsy_events += 1

                self._log_event(
                    "DROWSY_ALERT",
                    result,
                )

            if (
                looking_away
                and not person.was_looking_away
            ):

                person.distraction_events += 1

                self._log_event(
                    "DISTRACTION_START",
                    result,
                )

            if (
                not looking_away
                and person.was_looking_away
            ):

                self._log_event(
                    "DISTRACTION_END",
                    result,
                )

            # Log yawn exactly once
            if (
                is_yawning
                and person.mouth_open_streak
                == self.cfg.mar_consec_frames
            ):

                self._log_event(
                    "YAWN_DETECTED",
                    result,
                )

            # ----------------------------------------------------------
            # Save previous states
            # ----------------------------------------------------------

            person.was_closed = eyes_closed

            person.was_looking_away = looking_away

            # ----------------------------------------------------------
            # Save and publish result
            # ----------------------------------------------------------

            results.append(result)

            self._publish(result)

        return results

    # ------------------------------------------------------------------
    # Dashboard publishing
    # ------------------------------------------------------------------

    def _publish(
        self,
        result: FrameResult,
    ) -> None:

        if self.on_person_update is None:
            return

        self.on_person_update(
            result.person_id,
            {
                "timestamp": result.timestamp,
                "state": result.state,
                "score": result.attention_score,

                "ear": result.ear,
                "mar": result.mar,

                "eyes_closed": result.eyes_closed,
                "is_yawning": result.is_yawning,

                "yawn_count": result.yawn_count,

                "pitch": result.pitch,
                "yaw": result.yaw,
                "roll": result.roll,

                "calibration_progress":
                    result.calibration_progress,
            },
        )

    # ------------------------------------------------------------------
    # Active persons
    # ------------------------------------------------------------------

    def active_person_ids(self) -> list[int]:

        return list(
            self._persons.keys()
        )

    # ------------------------------------------------------------------
    # Session statistics
    # ------------------------------------------------------------------

    def session_stats(self) -> dict:

        return {
            pid: {
                "blink_count": p.blink_count,
                "yawn_count": p.yawn_count,
                "drowsy_events": p.drowsy_events,
                "distraction_events": p.distraction_events,
            }
            for pid, p in self._persons.items()
        }