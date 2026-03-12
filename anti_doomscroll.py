"""
anti_doomscroll.py - Catches you doom-scrolling and plays an annoying video.

Usage:
    python anti_doomscroll.py [options]

Options:
    --threshold FLOAT      Downward iris ratio to trigger (default: 0.62)
    --delay FLOAT          Seconds the full doomscroll pattern must persist before trigger
    --rearm-seconds FLOAT  Seconds of clear attention before rearming
    --calibration-seconds FLOAT
                            Seconds to learn your neutral look-ahead baseline
    --video PATH           Path to video file (default: annoying.mp4 beside script)
    --phone-model PATH     Path to phone detector model
    --phone-score FLOAT    Minimum confidence for phone detection
    --disable-phone-check  Disable phone detection and use gaze only
    --no-debug             Hide the webcam debug window
    --sensitivity low|medium|high  Preset sensitivity (overrides --threshold)
    --camera INT           Camera index to open (default: 0)
"""

import argparse
from collections import deque
from contextlib import ExitStack
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------
class _MissingModule:
    def __init__(self, name: str):
        self.name = name

    def __getattr__(self, attr: str) -> Any:
        raise RuntimeError(f"Missing optional dependency: {self.name}")


MISSING_PACKAGES: list[str] = []

def _import_optional(module_name: str, package_name: str) -> Any:
    try:
        return __import__(module_name)
    except ImportError:
        MISSING_PACKAGES.append(package_name)
        return _MissingModule(package_name)

 
cv2 = _import_optional("cv2", "opencv-python")
mp = _import_optional("mediapipe", "mediapipe")


def _require_cv2() -> Any:
    return cv2


def _require_mp() -> Any:
    return mp

# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------
DEFAULT_DOWNWARD_GAZE_SECONDS = 5.0
DEFAULT_REARM_SECONDS = 1.5
DEFAULT_DOWNWARD_IRIS_RATIO = 0.62
DEFAULT_CALIBRATION_SECONDS = 1.5
GAZE_HOLD_TOLERANCE_SECONDS = 0.25
PHONE_PRESENCE_HOLD_SECONDS = 0.75
RELATIVE_IRIS_DELTA = 0.035
RELATIVE_PITCH_DELTA = 0.025
PHONE_DIRECTION_COSINE_THRESHOLD = 0.65
SCROLL_MOTION_SCORE_THRESHOLD = 0.028
SCROLL_MOTION_WINDOW_SECONDS = 2.0
SCROLL_MOTION_MIN_EVENTS = 5
VIDEO_FILENAME = "annoying.mp4"
FACE_LANDMARKER_MODEL_FILENAME = "face_landmarker.task"
PHONE_DETECTOR_MODEL_FILENAME = "efficientdet_lite2.tflite"
PHONE_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite2/float32/latest/efficientdet_lite2.tflite"
)
DEBUG_WINDOW = "Anti Doomscroll (ESC to quit)"
TRIGGER_TEXT = "STOP SCROLLING!"
DEFAULT_PHONE_SCORE_THRESHOLD = 0.35
PHONE_CATEGORY_NAME = "cell phone"
PHONE_SCROLL_ZONE_Y_RATIO = 0.42
PHONE_MIN_AREA_RATIO = 0.0015
PHONE_DETECTION_INTERVAL_SECONDS = 0.18
FPS_SMOOTHING_WINDOW_SECONDS = 1.0

SENSITIVITY_PRESETS = {
    "low":    0.68,  # only very obvious downward gaze
    "medium": 0.62,  # balanced default
    "high":   0.56,  # triggers on slight downward glances
}

# Eye landmark indices (MediaPipe 478-point mesh)
LEFT_EYE  = {"upper": 159, "lower": 145, "iris": (468, 469, 470, 471)}
RIGHT_EYE = {"upper": 386, "lower": 374, "iris": (473, 474, 475, 476)}
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
NOSE_TIP = 1
CHIN = 152

# ---------------------------------------------------------------------------
# MediaPipe backend detection
# ---------------------------------------------------------------------------
MEDIAPIPE_BACKEND: str | None = None
MEDIAPIPE_ERROR:   str | None = None
mp_face_mesh   = None
mp_drawing     = None
mp_tasks_vision = None
mp_base_options = None

if mp is not None:
    try:
        solutions = getattr(mp, "solutions", None)
        tasks     = getattr(mp, "tasks", None)
        if (tasks is not None
              and hasattr(tasks, "vision")
              and hasattr(tasks, "BaseOptions")):
            mp_tasks_vision  = tasks.vision
            mp_base_options  = tasks.BaseOptions

        if solutions is not None and hasattr(solutions, "face_mesh"):
            MEDIAPIPE_BACKEND = "solutions"
            mp_face_mesh = solutions.face_mesh
            mp_drawing   = solutions.drawing_utils
        elif mp_tasks_vision is not None and mp_base_options is not None:
            MEDIAPIPE_BACKEND = "tasks"
        else:
            MEDIAPIPE_ERROR = (
                "Installed mediapipe exposes neither `solutions.face_mesh` "
                "nor `tasks.vision`."
            )
    except Exception as exc:
        MEDIAPIPE_ERROR = str(exc)


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

def validate_runtime() -> bool:
    if MISSING_PACKAGES:
        print("❌  Missing required packages:")
        for pkg in MISSING_PACKAGES:
            print(f"     pip install {pkg}")
        print("\nQuick fix:")
        print("     pip install -r requirements.txt")
        print("or:  pip install opencv-python mediapipe")
        return False

    if MEDIAPIPE_BACKEND is None:
        version = getattr(mp, "__version__", "unknown")
        print(f"❌  Unsupported mediapipe installation (version {version}).")
        if MEDIAPIPE_ERROR:
            print(f"   Detail: {MEDIAPIPE_ERROR}")
        return False

    if MEDIAPIPE_BACKEND == "tasks":
        if not (hasattr(mp, "Image") and hasattr(mp, "ImageFormat")):
            print("❌  This mediapipe Tasks build is missing Image/ImageFormat.")
            return False
        model_path = _resolve_asset(FACE_LANDMARKER_MODEL_FILENAME)
        if not model_path.exists():
            print(f"❌  Missing face landmarker model: {model_path}")
            print(
                "   Download it from:\n"
                "   https://storage.googleapis.com/mediapipe-models/"
                "   face_landmarker/face_landmarker/float16/latest/face_landmarker.task\n"
                "   and place it next to this script."
            )
            return False

    return True


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve_asset(filename: str) -> Path:
    return Path(__file__).resolve().parent / filename


# ---------------------------------------------------------------------------
# VLC detection & playback
# ---------------------------------------------------------------------------

def find_vlc() -> str | None:
    candidates: list[Path] = []

    hit = shutil.which("vlc")
    if hit:
        candidates.append(Path(hit))

    if sys.platform.startswith("win"):
        for env_var in ("ProgramFiles", "ProgramFiles(x86)"):
            root = os.environ.get(env_var)
            if root:
                candidates.append(Path(root) / "VideoLAN" / "VLC" / "vlc.exe")
    elif sys.platform == "darwin":
        candidates.append(Path("/Applications/VLC.app/Contents/MacOS/VLC"))

    for c in candidates:
        if c.exists():
            return str(c)
    return None


def play_video(vlc_path: str, video_path: Path) -> subprocess.Popen | None:
    if not vlc_path or not video_path.exists():
        return None
    try:
        return subprocess.Popen(
            [vlc_path, "--play-and-exit", "--fullscreen", str(video_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        print(f"⚠️  Could not launch VLC: {exc}")
        return None


def cleanup_player(proc: subprocess.Popen | None) -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()


def next_video_timestamp_ms(last_timestamp_ms: int | None) -> int:
    timestamp_ms = time.monotonic_ns() // 1_000_000
    if last_timestamp_ms is not None and timestamp_ms <= last_timestamp_ms:
        return last_timestamp_ms + 1
    return timestamp_ms


# ---------------------------------------------------------------------------
# Gaze detection helpers
# ---------------------------------------------------------------------------

def _eye_vertical_ratio(landmarks, eye: dict) -> float | None:
    upper_y = landmarks[eye["upper"]].y
    lower_y = landmarks[eye["lower"]].y
    height  = lower_y - upper_y
    if height <= 1e-4:
        return None
    iris_y = sum(landmarks[i].y for i in eye["iris"]) / len(eye["iris"])
    return (iris_y - upper_y) / height


def _average_landmark_y(landmarks, indices: tuple[int, ...]) -> float:
    return sum(landmarks[i].y for i in indices) / len(indices)


def _average_landmark_x(landmarks, indices: tuple[int, ...]) -> float:
    return sum(landmarks[i].x for i in indices) / len(indices)


def _head_pitch_ratio(landmarks) -> float | None:
    eye_center_y = (
        _average_landmark_y(landmarks, LEFT_EYE_CORNERS)
        + _average_landmark_y(landmarks, RIGHT_EYE_CORNERS)
    ) / 2
    chin_y = landmarks[CHIN].y
    face_height = chin_y - eye_center_y
    if face_height <= 1e-4:
        return None
    return (landmarks[NOSE_TIP].y - eye_center_y) / face_height


def gaze_metrics(landmarks) -> tuple[float | None, float | None]:
    ratios = [
        r for r in (
            _eye_vertical_ratio(landmarks, LEFT_EYE),
            _eye_vertical_ratio(landmarks, RIGHT_EYE),
        )
        if r is not None
    ]
    iris_ratio = (sum(ratios) / len(ratios)) if ratios else None
    return iris_ratio, _head_pitch_ratio(landmarks)


def is_gaze_down(
    iris_ratio: float | None,
    pitch_ratio: float | None,
    threshold: float,
    baseline_iris: float | None,
    baseline_pitch: float | None,
) -> bool:
    if iris_ratio is None:
        return (
            baseline_pitch is not None
            and pitch_ratio is not None
            and pitch_ratio >= baseline_pitch + (RELATIVE_PITCH_DELTA * 1.5)
        )

    absolute_hit = iris_ratio >= threshold
    relative_iris_hit = (
        baseline_iris is not None
        and iris_ratio >= baseline_iris + RELATIVE_IRIS_DELTA
    )
    combined_hit = (
        baseline_iris is not None
        and baseline_pitch is not None
        and pitch_ratio is not None
        and iris_ratio >= baseline_iris + (RELATIVE_IRIS_DELTA * 0.5)
        and pitch_ratio >= baseline_pitch + RELATIVE_PITCH_DELTA
    )
    return absolute_hit or relative_iris_hit or combined_hit


def face_pointing_toward_phone(
    landmarks,
    phone_detection,
    frame_shape: tuple[int, int, int],
    gaze_down: bool,
) -> tuple[bool, float | None]:
    if not gaze_down or phone_detection is None:
        return False, None

    frame_h, frame_w = frame_shape[:2]
    if frame_h <= 0 or frame_w <= 0:
        return False, None

    eye_center_x = (
        _average_landmark_x(landmarks, LEFT_EYE_CORNERS)
        + _average_landmark_x(landmarks, RIGHT_EYE_CORNERS)
    ) / 2
    eye_center_y = (
        _average_landmark_y(landmarks, LEFT_EYE_CORNERS)
        + _average_landmark_y(landmarks, RIGHT_EYE_CORNERS)
    ) / 2

    face_dx = landmarks[NOSE_TIP].x - eye_center_x
    face_dy = landmarks[NOSE_TIP].y - eye_center_y

    box = phone_detection.bounding_box
    phone_x = (box.origin_x + (box.width / 2)) / frame_w
    phone_y = (box.origin_y + (box.height / 2)) / frame_h
    target_dx = phone_x - eye_center_x
    target_dy = phone_y - eye_center_y

    face_len = (face_dx * face_dx + face_dy * face_dy) ** 0.5
    target_len = (target_dx * target_dx + target_dy * target_dy) ** 0.5
    if face_len <= 1e-4 or target_len <= 1e-4:
        return False, None

    cosine = ((face_dx * target_dx) + (face_dy * target_dy)) / (face_len * target_len)
    phone_below_face = phone_y > eye_center_y + 0.03
    return phone_below_face and cosine >= PHONE_DIRECTION_COSINE_THRESHOLD, cosine


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_text(frame, text: str, y_ratio: float = 0.15,
              color: tuple = (0, 0, 255)) -> None:
    cv = _require_cv2()
    font       = cv.FONT_HERSHEY_SIMPLEX
    scale      = 1.2
    thickness  = 3
    (tw, th), _ = cv.getTextSize(text, font, scale, thickness)
    x = int((frame.shape[1] - tw) / 2)
    y = max(int(frame.shape[0] * y_ratio), th + 10)
    # Shadow for readability
    cv.putText(frame, text, (x + 2, y + 2), font, scale,
               (0, 0, 0), thickness + 2, cv.LINE_AA)
    cv.putText(frame, text, (x, y), font, scale, color, thickness, cv.LINE_AA)


def draw_progress_bar(frame, progress: float) -> None:
    cv = _require_cv2()
    h, w = frame.shape[:2]
    bar_h = 10
    progress = min(max(progress, 0.0), 1.0)
    filled = int((w - 32) * progress)
    y1 = h - 24
    y2 = y1 + bar_h
    cv.rectangle(frame, (16, y1), (w - 16, y2), (38, 42, 52), -1)
    if filled > 0:
        cv.rectangle(frame, (16, y1), (16 + filled, y2), (0, 165, 255), -1)
    cv.rectangle(frame, (16, y1), (w - 16, y2), (88, 98, 120), 1)


def format_metric(value, fmt: str) -> str:
    return "n/a" if value is None else format(value, fmt)


def draw_panel(
    frame,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    fill_color: tuple[int, int, int],
    border_color: tuple[int, int, int],
    alpha: float = 0.72,
) -> None:
    cv = _require_cv2()
    x1, y1 = top_left
    x2, y2 = bottom_right
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, frame.shape[1])
    y2 = min(y2, frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    overlay[:] = fill_color
    cv.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)
    cv.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), border_color, 1, cv.LINE_AA)


def draw_badge(
    frame,
    text: str,
    origin: tuple[int, int],
    fill_color: tuple[int, int, int],
    text_color: tuple[int, int, int] = (245, 245, 245),
) -> None:
    cv = _require_cv2()
    x, y = origin
    font = cv.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (tw, th), _ = cv.getTextSize(text, font, scale, thickness)
    pad_x = 8
    pad_y = 6
    cv.rectangle(
        frame,
        (x, y - th - pad_y),
        (x + tw + (pad_x * 2), y + pad_y),
        fill_color,
        -1,
        cv.LINE_AA,
    )
    cv.putText(
        frame,
        text,
        (x + pad_x, y),
        font,
        scale,
        text_color,
        thickness,
        cv.LINE_AA,
    )


def draw_debug_hud(
    frame,
    *,
    mode_label: str,
    state_label: str,
    state_color: tuple[int, int, int],
    banner_text: str | None,
    banner_color: tuple[int, int, int],
    progress_value: float | None,
    progress_label: str | None,
    fps: float,
    frame_count: int,
    iris_ratio: float | None,
    baseline_iris: float | None,
    threshold: float,
    pitch_ratio: float | None,
    gaze_down: bool,
    phone_check_enabled: bool,
    phone_present: bool,
    phone_score: float,
    phone_detection,
    face_toward_phone: bool,
    direction_cosine: float | None,
    motion_score: float,
    scrolling_evidence: bool,
) -> None:
    cv = _require_cv2()
    h, w = frame.shape[:2]
    panel_w = min(max(420, w - 32), 660)
    top_left = (16, 16)
    bottom_right = (min(16 + panel_w, w - 16), min(156, h - 56))
    draw_panel(frame, top_left, bottom_right, (18, 23, 31), (84, 96, 118), alpha=0.74)

    cv.putText(
        frame,
        "ANTI DOOMSCROLL",
        (32, 44),
        cv.FONT_HERSHEY_SIMPLEX,
        0.78,
        (240, 244, 248),
        2,
        cv.LINE_AA,
    )
    cv.putText(
        frame,
        f"Mode: {mode_label}",
        (32, 72),
        cv.FONT_HERSHEY_SIMPLEX,
        0.54,
        (188, 202, 216),
        1,
        cv.LINE_AA,
    )
    draw_badge(frame, state_label.upper(), (32, 100), state_color)

    perf_panel_w = 166
    draw_panel(
        frame,
        (w - perf_panel_w - 16, 16),
        (w - 16, 74),
        (18, 23, 31),
        (84, 96, 118),
        alpha=0.74,
    )
    cv.putText(
        frame,
        f"FPS {fps:4.1f}",
        (w - perf_panel_w, 42),
        cv.FONT_HERSHEY_SIMPLEX,
        0.62,
        (240, 244, 248),
        2,
        cv.LINE_AA,
    )
    cv.putText(
        frame,
        f"Frame {frame_count}",
        (w - perf_panel_w, 64),
        cv.FONT_HERSHEY_SIMPLEX,
        0.48,
        (188, 202, 216),
        1,
        cv.LINE_AA,
    )

    metric_y = 126
    cv.putText(
        frame,
        (
            f"Iris {format_metric(iris_ratio, '.3f')}   "
            f"Base {format_metric(baseline_iris, '.3f')}   "
            f"Thr {threshold:.3f}   Pitch {format_metric(pitch_ratio, '.3f')}"
        ),
        (32, metric_y),
        cv.FONT_HERSHEY_PLAIN,
        1.2,
        (224, 230, 236),
        1,
        cv.LINE_AA,
    )
    cv.putText(
        frame,
        (
            f"Gaze {'down' if gaze_down else 'clear'}   "
            f"Phone {'off' if not phone_check_enabled else ('yes' if phone_present else 'no')}   "
            f"Score {format_metric(phone_score if phone_detection is not None else None, '.2f')}   "
            f"Toward {'yes' if face_toward_phone else 'no'}"
        ),
        (32, metric_y + 20),
        cv.FONT_HERSHEY_PLAIN,
        1.2,
        (224, 230, 236),
        1,
        cv.LINE_AA,
    )
    cv.putText(
        frame,
        (
            f"Dir {format_metric(direction_cosine, '.2f')}   "
            f"Motion {motion_score:.3f}   "
            f"Scroll {'yes' if scrolling_evidence else 'no'}   ESC quit"
        ),
        (32, metric_y + 40),
        cv.FONT_HERSHEY_PLAIN,
        1.2,
        (224, 230, 236),
        1,
        cv.LINE_AA,
    )

    if banner_text:
        draw_text(frame, banner_text, 0.14, banner_color)

    if progress_value is not None:
        draw_progress_bar(frame, progress_value)
        if progress_label:
            cv.putText(
                frame,
                progress_label,
                (24, h - 32),
                cv.FONT_HERSHEY_PLAIN,
                1.2,
                (224, 230, 236),
                1,
                cv.LINE_AA,
            )


def create_phone_detector(model_path: Path, score_threshold: float):
    options = mp_tasks_vision.ObjectDetectorOptions(
        base_options=mp_base_options(model_asset_path=str(model_path)),
        running_mode=mp_tasks_vision.RunningMode.VIDEO,
        max_results=1,
        score_threshold=score_threshold,
        category_allowlist=[PHONE_CATEGORY_NAME],
    )
    return mp_tasks_vision.ObjectDetector.create_from_options(options)


def detect_phone(phone_detector, rgb_frame, timestamp_ms: int):
    mp_runtime = _require_mp()
    mp_img = mp_runtime.Image(image_format=mp_runtime.ImageFormat.SRGB, data=rgb_frame)
    results = phone_detector.detect_for_video(mp_img, timestamp_ms)
    best_detection = None
    best_score = 0.0

    for detection in results.detections:
        for category in detection.categories:
            category_name = (category.category_name or category.display_name or "").lower()
            score = category.score or 0.0
            if category_name == PHONE_CATEGORY_NAME and score > best_score:
                best_detection = detection
                best_score = score

    return best_detection, best_score


def phone_detection_in_scroll_zone(detection, frame_shape: tuple[int, int, int]) -> bool:
    frame_h, frame_w = frame_shape[:2]
    box = detection.bounding_box
    center_y_ratio = (box.origin_y + (box.height / 2)) / max(frame_h, 1)
    area_ratio = (box.width * box.height) / max(frame_h * frame_w, 1)
    return center_y_ratio >= PHONE_SCROLL_ZONE_Y_RATIO and area_ratio >= PHONE_MIN_AREA_RATIO


def draw_phone_detection(frame, detection, score: float, active: bool) -> None:
    cv = _require_cv2()
    box = detection.bounding_box
    x1 = max(box.origin_x, 0)
    y1 = max(box.origin_y, 0)
    x2 = min(box.origin_x + box.width, frame.shape[1] - 1)
    y2 = min(box.origin_y + box.height, frame.shape[0] - 1)
    color = (0, 255, 255) if active else (0, 140, 255)
    cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv.putText(
        frame,
        f"phone {score:.2f}",
        (x1, max(y1 - 8, 18)),
        cv.FONT_HERSHEY_PLAIN,
        1.2,
        color,
        2,
    )


def extract_phone_roi_gray(frame, detection):
    cv = _require_cv2()
    box = detection.bounding_box
    x1 = max(box.origin_x, 0)
    y1 = max(box.origin_y, 0)
    x2 = min(box.origin_x + box.width, frame.shape[1])
    y2 = min(box.origin_y + box.height, frame.shape[0])
    if x2 - x1 < 24 or y2 - y1 < 24:
        return None

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    return cv.resize(gray, (96, 160))


def phone_scroll_motion_score(previous_roi, current_roi) -> float:
    cv = _require_cv2()
    if previous_roi is None or current_roi is None:
        return 0.0
    diff = cv.absdiff(previous_roi, current_roi)
    return float(diff.mean() / 255.0)


# ---------------------------------------------------------------------------
# MediaPipe face tracker factory + frame processor
# ---------------------------------------------------------------------------

def create_face_tracker():
    if MEDIAPIPE_BACKEND == "solutions":
        return mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
        )

    options = mp_tasks_vision.FaceLandmarkerOptions(
        base_options=mp_base_options(
            model_asset_path=str(_resolve_asset(FACE_LANDMARKER_MODEL_FILENAME))
        ),
        running_mode=mp_tasks_vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_tasks_vision.FaceLandmarker.create_from_options(options)


def process_frame(face_tracker, frame, rgb_frame, timestamp_ms: int, draw_landmarks: bool):
    """Return landmark list, or None if no face detected."""
    if MEDIAPIPE_BACKEND == "solutions":
        results = face_tracker.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None
        fl = results.multi_face_landmarks[0]
        if draw_landmarks:
            mp_drawing.draw_landmarks(
                frame, fl, mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 180, 255), thickness=1),
            )
        return fl.landmark

    mp_runtime = _require_mp()
    mp_img  = mp_runtime.Image(image_format=mp_runtime.ImageFormat.SRGB, data=rgb_frame)
    results = face_tracker.detect_for_video(mp_img, timestamp_ms)
    if not results.face_landmarks:
        return None
    return results.face_landmarks[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Anti doomscroll - catches downward gaze and plays a video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--threshold", type=float, default=None,
                   help=f"Iris ratio threshold (default {DEFAULT_DOWNWARD_IRIS_RATIO})")
    p.add_argument("--delay", type=float, default=DEFAULT_DOWNWARD_GAZE_SECONDS,
                   help="Seconds the full doomscroll pattern must persist before trigger")
    p.add_argument("--rearm-seconds", "--cooldown", dest="rearm_seconds",
                   type=float, default=DEFAULT_REARM_SECONDS,
                   help="Seconds of clear attention before the next trigger")
    p.add_argument("--calibration-seconds", type=float, default=DEFAULT_CALIBRATION_SECONDS,
                   help="Seconds to calibrate while you look straight ahead")
    p.add_argument("--video", type=Path, default=None,
                   help="Path to video file")
    p.add_argument("--phone-model", type=Path, default=None,
                   help="Path to MediaPipe object detector model for phones")
    p.add_argument("--phone-score", type=float, default=DEFAULT_PHONE_SCORE_THRESHOLD,
                   help="Minimum confidence for phone detection")
    p.add_argument("--disable-phone-check", action="store_true",
                   help="Disable phone detection and use gaze only")
    p.add_argument("--no-debug", action="store_true",
                   help="Hide webcam window")
    p.add_argument("--sensitivity", choices=SENSITIVITY_PRESETS,
                   help="Sensitivity preset (overrides --threshold)")
    p.add_argument("--camera", type=int, default=0,
                   help="Camera index (default 0)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    cv = _require_cv2()
    args = parse_args()
    args.delay = max(args.delay, 0.0)
    args.rearm_seconds = max(args.rearm_seconds, 0.0)
    args.calibration_seconds = max(args.calibration_seconds, 0.0)
    args.phone_score = min(max(args.phone_score, 0.0), 1.0)
    show_debug = not args.no_debug

    if not validate_runtime():
        return 1

    # Resolve threshold
    if args.sensitivity:
        threshold = SENSITIVITY_PRESETS[args.sensitivity]
    elif args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = DEFAULT_DOWNWARD_IRIS_RATIO

    video_path = args.video or _resolve_asset(VIDEO_FILENAME)
    if not video_path.exists():
        print(f"⚠️  Video not found: {video_path}  (text overlay only)")

    vlc_path = find_vlc()
    if vlc_path is None:
        print("⚠️  VLC not found — install it for video playback. Text overlay only.")
    else:
        print(f"✅  VLC found: {vlc_path}")

    phone_model_path = args.phone_model or _resolve_asset(PHONE_DETECTOR_MODEL_FILENAME)
    phone_check_enabled = False
    phone_detector_ctx = None
    if args.disable_phone_check:
        print("Phone check: disabled; using gaze only.")
    elif mp_tasks_vision is None or mp_base_options is None:
        print("Phone check: unavailable in this mediapipe build; using gaze only.")
    elif not phone_model_path.exists():
        print(f"Phone check: model not found at {phone_model_path}")
        print(
            "Download the official MediaPipe object detector model:\n"
            f"{PHONE_DETECTOR_MODEL_URL}\n"
            f"and place it next to the script as {PHONE_DETECTOR_MODEL_FILENAME}."
        )
    else:
        try:
            phone_detector_ctx = create_phone_detector(phone_model_path, args.phone_score)
            phone_check_enabled = True
            print(
                "Phone check: enabled "
                f"({phone_model_path.name}, score >= {args.phone_score:.2f})"
            )
        except Exception as exc:
            print(f"Phone check init failed: {exc}")
            print("Falling back to gaze only.")

    cap = cv.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌  Could not open camera index {args.camera}.")
        return 1

    # Attempt higher resolution
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    print(f"\nThreshold: {threshold}  |  Persistence: {args.delay}s  |  "
          f"Rearm: {args.rearm_seconds}s")
    if args.calibration_seconds > 0:
        print(f"Calibration: {args.calibration_seconds}s looking straight ahead")
    if show_debug:
        print("Press ESC in the debug window (or Ctrl-C) to quit.\n")
    else:
        print("Running headless. Press Ctrl-C to quit.\n")

    try:
        face_tracker_ctx = create_face_tracker()
    except Exception as exc:
        cap.release()
        print(f"❌  Failed to init face tracker: {exc}")
        return 1

    candidate_started_at: float | None = None
    clear_started_at: float | None = None
    episode_triggered = False
    player_process: subprocess.Popen | None = None
    frame_count = 0
    last_timestamp_ms: int | None = None
    calibration_started_at: float | None = None
    calibration_iris_samples: list[float] = []
    calibration_pitch_samples: list[float] = []
    baseline_iris: float | None = None
    baseline_pitch: float | None = None
    last_positive_gaze_at: float | None = None
    last_positive_phone_at: float | None = None
    previous_phone_roi = None
    motion_event_times = deque()
    last_phone_detection_at: float | None = None
    cached_phone_detection = None
    cached_phone_score = 0.0
    fps_times = deque()

    try:
        with ExitStack() as stack:
            face_tracker = stack.enter_context(face_tracker_ctx)
            phone_detector = (
                stack.enter_context(phone_detector_ctx)
                if phone_detector_ctx is not None
                else None
            )
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌  Webcam read failed.")
                    return 1

                frame     = cv.flip(frame, 1)
                rgb       = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                now       = time.monotonic()
                ts_ms     = next_video_timestamp_ms(last_timestamp_ms)
                last_timestamp_ms = ts_ms
                frame_count += 1
                fps = 0.0
                if show_debug:
                    fps_times.append(now)
                    while fps_times and (now - fps_times[0]) > FPS_SMOOTHING_WINDOW_SECONDS:
                        fps_times.popleft()
                    fps = len(fps_times) / max(FPS_SMOOTHING_WINDOW_SECONDS, 1e-6)

                # Clean up finished player
                if player_process is not None and player_process.poll() is not None:
                    player_process = None

                landmarks = process_frame(face_tracker, frame, rgb, ts_ms, draw_landmarks=show_debug)
                phone_detection = None
                phone_score = 0.0
                raw_phone_present = not phone_check_enabled
                iris_ratio: float | None = None
                pitch_ratio: float | None = None
                direction_cosine: float | None = None
                motion_score = 0.0
                raw_gaze_down = False
                face_toward_phone = False
                scrolling_evidence = False
                calibrating = False
                banner_text: str | None = None
                banner_color = (0, 165, 255)
                progress_value: float | None = None
                progress_label: str | None = None
                state_label = "armed"
                state_color = (54, 136, 201)
                candidate_label = "Phone focus" if phone_check_enabled else "Downward gaze"

                if phone_detector is not None:
                    if (
                        last_phone_detection_at is None
                        or (now - last_phone_detection_at) >= PHONE_DETECTION_INTERVAL_SECONDS
                    ):
                        cached_phone_detection, cached_phone_score = detect_phone(phone_detector, rgb, ts_ms)
                        last_phone_detection_at = now
                    phone_detection = cached_phone_detection
                    phone_score = cached_phone_score
                    if phone_detection is not None:
                        raw_phone_present = phone_detection_in_scroll_zone(phone_detection, frame.shape)
                        if show_debug:
                            draw_phone_detection(frame, phone_detection, phone_score, raw_phone_present)

                if raw_phone_present:
                    last_positive_phone_at = now

                phone_present = raw_phone_present or (
                    last_positive_phone_at is not None
                    and (now - last_positive_phone_at) <= PHONE_PRESENCE_HOLD_SECONDS
                )

                current_phone_roi = None
                if phone_detection is not None and raw_phone_present:
                    current_phone_roi = extract_phone_roi_gray(frame, phone_detection)
                    if previous_phone_roi is not None and current_phone_roi is not None:
                        motion_score = phone_scroll_motion_score(previous_phone_roi, current_phone_roi)
                        if motion_score >= SCROLL_MOTION_SCORE_THRESHOLD:
                            motion_event_times.append(now)
                    previous_phone_roi = current_phone_roi
                else:
                    previous_phone_roi = None

                while motion_event_times and (now - motion_event_times[0]) > SCROLL_MOTION_WINDOW_SECONDS:
                    motion_event_times.popleft()

                if not phone_present:
                    motion_event_times.clear()

                if landmarks is not None:
                    iris_ratio, pitch_ratio = gaze_metrics(landmarks)

                    if args.calibration_seconds > 0 and baseline_iris is None and iris_ratio is not None:
                        if calibration_started_at is None:
                            calibration_started_at = now

                        elapsed_calibration = now - calibration_started_at
                        if elapsed_calibration < args.calibration_seconds:
                            calibrating = True
                            calibration_iris_samples.append(iris_ratio)
                            if pitch_ratio is not None:
                                calibration_pitch_samples.append(pitch_ratio)
                        elif calibration_iris_samples:
                            baseline_iris = statistics.median(calibration_iris_samples)
                            if calibration_pitch_samples:
                                baseline_pitch = statistics.median(calibration_pitch_samples)

                    if not calibrating:
                        raw_gaze_down = is_gaze_down(
                            iris_ratio,
                            pitch_ratio,
                            threshold,
                            baseline_iris,
                            baseline_pitch,
                        )

                if raw_gaze_down:
                    last_positive_gaze_at = now

                gaze_down = raw_gaze_down or (
                    last_positive_gaze_at is not None
                    and (now - last_positive_gaze_at) <= GAZE_HOLD_TOLERANCE_SECONDS
                )
                if landmarks is not None and phone_detection is not None and phone_present:
                    face_toward_phone, direction_cosine = face_pointing_toward_phone(
                        landmarks,
                        phone_detection,
                        frame.shape,
                        gaze_down,
                    )
                if phone_check_enabled:
                    scrolling_evidence = len(motion_event_times) >= SCROLL_MOTION_MIN_EVENTS
                    doomscroll_candidate = phone_present and face_toward_phone
                else:
                    scrolling_evidence = True
                    doomscroll_candidate = gaze_down
                should_trigger = doomscroll_candidate and scrolling_evidence

                # --- State machine ---
                if calibrating:
                    state_label = "calibrating"
                    state_color = (0, 140, 255)
                    candidate_started_at = None
                    clear_started_at = None
                    episode_triggered = False
                    previous_phone_roi = None
                    motion_event_times.clear()
                    remaining_calibration = max(
                        0.0,
                        args.calibration_seconds - (now - calibration_started_at),
                    )
                    banner_text = f"Look straight ahead {remaining_calibration:.1f}s"
                    banner_color = (255, 200, 0)
                    if args.calibration_seconds > 0:
                        elapsed_calibration = args.calibration_seconds - remaining_calibration
                        progress_value = min(elapsed_calibration / args.calibration_seconds, 1.0)
                        progress_label = "Calibrating baseline"
                else:
                    if doomscroll_candidate:
                        state_label = "tracking"
                        state_color = (0, 165, 255)
                        clear_started_at = None
                        if candidate_started_at is None:
                            candidate_started_at = now

                        elapsed = now - candidate_started_at
                        remaining = max(0.0, args.delay - elapsed)

                        if not episode_triggered:
                            if remaining > 0:
                                banner_text = f"{candidate_label} {remaining:.1f}s"
                                banner_color = (0, 165, 255)
                                if args.delay > 0:
                                    progress_value = min(elapsed / args.delay, 1.0)
                                else:
                                    progress_value = 1.0
                                progress_label = "Sustained attention pattern"
                            elif not scrolling_evidence:
                                state_label = "verifying"
                                state_color = (0, 165, 255)
                                banner_text = "Need scrolling motion"
                                progress_value = min(
                                    len(motion_event_times) / SCROLL_MOTION_MIN_EVENTS,
                                    1.0,
                                )
                                progress_label = "Waiting for scrolling evidence"
                            else:
                                state_label = "triggered"
                                state_color = (0, 0, 255)
                                banner_text = TRIGGER_TEXT
                                banner_color = (0, 0, 255)
                                progress_value = 1.0
                                progress_label = "Trigger fired"
                                if player_process is None:
                                    player_process = play_video(vlc_path, video_path)
                                episode_triggered = True
                        else:
                            state_label = "triggered"
                            state_color = (0, 0, 255)
                            banner_text = "Refocus on your work"
                            banner_color = (0, 180, 255)
                    else:
                        candidate_started_at = None
                        if episode_triggered:
                            state_label = "rearming"
                            state_color = (255, 210, 0)
                            if clear_started_at is None:
                                clear_started_at = now
                            elif (now - clear_started_at) >= args.rearm_seconds:
                                episode_triggered = False
                        else:
                            clear_started_at = None

                    if landmarks is None and not episode_triggered:
                        state_label = "no face"
                        state_color = (110, 110, 110)
                        banner_text = "Face not detected"
                        banner_color = (180, 180, 180)
                    elif phone_check_enabled and gaze_down and not phone_present:
                        state_label = "searching"
                        state_color = (255, 200, 0)
                        banner_text = "Phone not detected"
                        banner_color = (255, 200, 0)
                    elif phone_check_enabled and phone_present and gaze_down and not face_toward_phone:
                        state_label = "off-axis"
                        state_color = (255, 200, 0)
                        banner_text = "Not looking at phone"
                        banner_color = (255, 200, 0)

                    if episode_triggered and not should_trigger and clear_started_at is not None:
                        secs_left = max(0.0, args.rearm_seconds - (now - clear_started_at))
                        banner_text = f"Rearming {secs_left:.1f}s"
                        banner_color = (255, 255, 0)
                        if args.rearm_seconds > 0:
                            progress_value = min((now - clear_started_at) / args.rearm_seconds, 1.0)
                            progress_label = "Clearing trigger state"

                if show_debug:
                    draw_debug_hud(
                        frame,
                        mode_label="phone + gaze" if phone_check_enabled else "gaze only",
                        state_label=state_label,
                        state_color=state_color,
                        banner_text=banner_text,
                        banner_color=banner_color,
                        progress_value=progress_value,
                        progress_label=progress_label,
                        fps=fps,
                        frame_count=frame_count,
                        iris_ratio=iris_ratio,
                        baseline_iris=baseline_iris,
                        threshold=threshold,
                        pitch_ratio=pitch_ratio,
                        gaze_down=gaze_down,
                        phone_check_enabled=phone_check_enabled,
                        phone_present=phone_present,
                        phone_score=phone_score,
                        phone_detection=phone_detection,
                        face_toward_phone=face_toward_phone,
                        direction_cosine=direction_cosine,
                        motion_score=motion_score,
                        scrolling_evidence=scrolling_evidence,
                    )
                    cv.imshow(DEBUG_WINDOW, frame)
                    if cv.waitKey(1) & 0xFF == 27:
                        print("Exiting…")
                        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        cleanup_player(player_process)
        cv.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
