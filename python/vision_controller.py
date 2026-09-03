"""
Vision HID Controller - vision-only prototype.

This version tracks a target and prints debug commands. When --serial-port is
provided, it also sends the commands to a compatible Arduino HID board:

    MOVE 12 -5
    STOP
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import serial
from serial.tools import list_ports
from ultralytics import YOLO


# Beginner-friendly configuration values. Tweak these first.
TARGET_CLASS = "person"
MIN_CONFIDENCE = 0.60
DEADZONE = 20
SENSITIVITY = 0.08
MAX_SPEED = 40
SMOOTHING_ALPHA = 0.25
LOST_TIMEOUT = 0.5
MIN_LOCK_MATCH_DISTANCE = 80
SIMILAR_DISTANCE_PIXELS = 10
SERIAL_BAUD = 115200
SERIAL_SEND_INTERVAL = 0.05
ARM_REFRESH_INTERVAL = 0.25
ARDUINO_STARTUP_DELAY = 2.0
PAUSE_HOTKEY_VK = 0x77  # Windows virtual-key code for F8.

REQUESTED_WIDTH = 1280
REQUESTED_HEIGHT = 720
REQUESTED_FPS = 60
CAPTURE_MODES = [
    (1280, 720),
    (960, 540),
    (854, 480),
    (640, 360),
    (800, 600),
    (640, 480),
    (424, 240),
    (320, 240),
]
INFERENCE_SIZE = 320
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

CAMERA_MEMORY_FILE = Path(__file__).with_name("camera_selection.json")
WINDOW_NAME = "Vision HID Controller"
NO_TARGET = "NO_TARGET"
LOCKED = "LOCKED"
LOST = "LOST"


class CursorPoint(ctypes.Structure):
    """Windows POINT structure used to read the current cursor position."""

    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class GlobalPauseHotkey:
    """Watch F8 globally and expose a thread-safe paused flag."""

    def __init__(self) -> None:
        self.paused = threading.Event()
        self.stop_requested = threading.Event()
        self.thread = threading.Thread(target=self._watch_key, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_requested.set()
        self.thread.join(timeout=0.2)

    def _watch_key(self) -> None:
        key_was_down = False

        while not self.stop_requested.is_set():
            key_is_down = bool(ctypes.windll.user32.GetAsyncKeyState(PAUSE_HOTKEY_VK) & 0x8000)

            if key_is_down and not key_was_down:
                if self.paused.is_set():
                    self.paused.clear()
                    print("Tracker resumed (F8)")
                else:
                    self.paused.set()
                    print("Tracker paused (F8)")

            key_was_down = key_is_down
            time.sleep(0.02)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision HID Controller")
    parser.add_argument(
        "--source",
        default=None,
        help="Camera index, video path, or stream URL. If omitted, you can choose from available cameras.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics YOLO model path or name. yolov8n.pt is small and quick.",
    )
    parser.add_argument(
        "--target-class",
        default=TARGET_CLASS,
        help="Only track detections with this YOLO class name.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=MIN_CONFIDENCE,
        help="Minimum confidence for tracked detections.",
    )
    parser.add_argument(
        "--deadzone",
        type=int,
        default=DEADZONE,
        help="Ignore raw pixel error smaller than this value.",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=SENSITIVITY,
        help="Scale raw pixel error into simulated mouse movement.",
    )
    parser.add_argument(
        "--max-speed",
        type=int,
        default=MAX_SPEED,
        help="Clamp each movement axis to this maximum absolute value.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=SMOOTHING_ALPHA,
        help="Low-pass filter amount from 0.0 to 1.0. Lower is smoother.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=REQUESTED_FPS,
        help="Requested camera capture FPS.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=REQUESTED_WIDTH,
        help="Preferred camera capture width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=REQUESTED_HEIGHT,
        help="Preferred camera capture height.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=INFERENCE_SIZE,
        help="YOLO inference image size. Smaller is faster but less precise.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional YOLO device, such as cpu, cuda, or 0. Leave unset for auto.",
    )
    parser.add_argument(
        "--serial-port",
        default="COM6",
        help="Arduino serial port. Defaults to COM6.",
    )
    parser.add_argument(
        "--no-arduino",
        action="store_const",
        const=None,
        dest="serial_port",
        help="Run in simulation mode without opening an Arduino serial port.",
    )
    parser.add_argument(
        "--serial-baud",
        type=int,
        default=SERIAL_BAUD,
        help="Arduino serial baud rate.",
    )
    parser.add_argument(
        "--list-serial-ports",
        action="store_true",
        help="List serial ports and exit without opening the camera.",
    )
    return parser.parse_args()


def print_serial_ports() -> None:
    """Show serial devices so the Arduino port is easy to identify."""
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return

    print("Available serial ports:")
    for port in ports:
        print(f"  {port.device}: {port.description}")


class ArduinoController:
    """Small line-based serial connection for the Arduino HID sketch."""

    def __init__(self, port: str, baud: int) -> None:
        print(f"Opening Arduino on {port} at {baud} baud...")
        self.connection = serial.Serial(port, baud, timeout=0.05, write_timeout=0.1)

        # Leonardo/Micro boards commonly reset when their serial port opens.
        time.sleep(ARDUINO_STARTUP_DELAY)
        self.connection.reset_input_buffer()
        self.connection.reset_output_buffer()
        self.last_command = ""
        self.last_send_time = 0.0
        self.last_arm_time = time.perf_counter()
        self.write_line("ARM")
        print("Arduino HID armed. Press q or Esc to stop and disarm it.")

    def write_line(self, command: str) -> None:
        """Send one newline-terminated ASCII command."""
        self.connection.write((command + "\n").encode("ascii"))
        self.connection.flush()

    def send_command(self, command: str) -> None:
        """Send changes immediately and repeat commands as a watchdog heartbeat."""
        current_time = time.perf_counter()

        # A camera or inference call can briefly stall while applications switch.
        # Re-arm periodically so control recovers after the Arduino watchdog fires.
        if current_time - self.last_arm_time >= ARM_REFRESH_INTERVAL:
            self.write_line("ARM")
            self.last_arm_time = current_time

        should_send = (
            command != self.last_command
            or current_time - self.last_send_time >= SERIAL_SEND_INTERVAL
        )
        if not should_send:
            return

        self.write_line(command)
        self.last_command = command
        self.last_send_time = current_time

        # Drain short Arduino acknowledgements so the receive buffer cannot grow.
        if self.connection.in_waiting:
            self.connection.read(self.connection.in_waiting)

    def close(self) -> None:
        """Stop movement, disarm HID, and close the port."""
        if not self.connection.is_open:
            return

        try:
            try:
                self.write_line("STOP")
                self.write_line("DISARM")
            except serial.SerialException:
                # The Arduino watchdog will disarm if the cable was unplugged.
                pass
        finally:
            self.connection.close()


def clamp(value: int, minimum: int, maximum: int) -> int:
    """Keep a number inside a safe range."""
    return max(minimum, min(maximum, value))


def get_desktop_cursor_position() -> tuple[int, int]:
    """Return the current Windows cursor position in primary-screen pixels."""
    point = CursorPoint()
    if not ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
        raise RuntimeError("Windows could not read the current cursor position.")
    return (point.x, point.y)


def map_frame_point_to_desktop(
    frame_point: tuple[int, int],
    frame_size: tuple[int, int],
) -> tuple[int, int]:
    """Map a point in the camera/OBS frame onto the primary desktop."""
    frame_width, frame_height = frame_size
    desktop_width = ctypes.windll.user32.GetSystemMetrics(0)
    desktop_height = ctypes.windll.user32.GetSystemMetrics(1)

    if frame_width <= 1 or frame_height <= 1 or desktop_width <= 0 or desktop_height <= 0:
        raise RuntimeError("Invalid frame or desktop dimensions.")

    desktop_x = round(frame_point[0] * (desktop_width - 1) / (frame_width - 1))
    desktop_y = round(frame_point[1] * (desktop_height - 1) / (frame_height - 1))
    return (
        clamp(desktop_x, 0, desktop_width - 1),
        clamp(desktop_y, 0, desktop_height - 1),
    )


def camera_label(index: int, width: int, height: int) -> str:
    """Return a simple label for a camera discovered by OpenCV."""
    return f"Camera {index}: {width}x{height}"


def discover_cameras(max_index: int = 10) -> list[dict[str, int]]:
    """Try camera indexes and return the ones that can provide a frame."""
    cameras = []

    for index in range(max_index):
        cap = open_camera_index(index)

        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                height, width = frame.shape[:2]
                cameras.append({"index": index, "width": width, "height": height})

        cap.release()

    return cameras


def open_camera_index(index: int) -> cv2.VideoCapture:
    """Open a numeric camera index with a stable backend for the current OS."""
    if os.name == "nt":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)

    return cv2.VideoCapture(index)


def load_remembered_camera() -> int | None:
    """Read the last chosen camera index from a tiny JSON file."""
    if not CAMERA_MEMORY_FILE.exists():
        return None

    try:
        data = json.loads(CAMERA_MEMORY_FILE.read_text(encoding="utf-8"))
        return int(data["camera_index"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def remember_camera(index: int) -> None:
    """Save the camera index so the next run can offer it as the default."""
    CAMERA_MEMORY_FILE.write_text(
        json.dumps({"camera_index": index}, indent=2),
        encoding="utf-8",
    )


def choose_camera(cameras: list[dict[str, int]]) -> int:
    """Print available cameras and ask the user which one to use."""
    if not cameras:
        raise RuntimeError("No cameras were found.")

    remembered = load_remembered_camera()
    available_indexes = {camera["index"] for camera in cameras}

    print("Available cameras:")
    for camera in cameras:
        marker = " (remembered)" if camera["index"] == remembered else ""
        print(
            f"  {camera['index']}: "
            f"{camera_label(camera['index'], camera['width'], camera['height'])}"
            f"{marker}"
        )

    default_index = remembered if remembered in available_indexes else cameras[0]["index"]
    while True:
        choice = input(f"Choose camera index [{default_index}]: ").strip()

        if not choice:
            selected_index = default_index
        else:
            try:
                selected_index = int(choice)
            except ValueError:
                print("Please enter a camera number.")
                continue

        if selected_index in available_indexes:
            break

        print(f"Camera {selected_index} is not available.")

    remember_camera(selected_index)
    return selected_index


def open_camera(
    source: str | None,
    preferred_width: int,
    preferred_height: int,
    requested_fps: int,
) -> cv2.VideoCapture:
    """
    Open a camera, video file, or stream.

    Numeric sources use OpenCV camera indexes. When source is None, available
    cameras are printed and the user chooses one from the console.
    """
    if source is None:
        cameras = discover_cameras()
        camera_index = choose_camera(cameras)
        cap = open_camera_index(camera_index)
    elif source.isdigit():
        camera_index = int(source)
        remember_camera(camera_index)
        cap = open_camera_index(camera_index)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    configure_capture(cap, preferred_width, preferred_height, requested_fps)
    return cap


def configure_capture(
    cap: cv2.VideoCapture,
    preferred_width: int,
    preferred_height: int,
    requested_fps: int,
) -> None:
    """
    Request the preferred size at 60 FPS, then try smaller 60 FPS-friendly modes.

    Many webcams need MJPG mode to reach 60 FPS. OpenCV still depends on the
    camera driver, so we test common sizes and keep the one with the best
    reported FPS. If FPS ties, we keep the larger image.
    """
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    candidates = [(preferred_width, preferred_height)]
    for mode in CAPTURE_MODES:
        if mode not in candidates:
            candidates.append(mode)

    best_mode = None
    best_score = None

    for width, height in candidates:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, requested_fps)

        ok, frame = cap.read()
        if not ok:
            continue

        actual_height, actual_width = frame.shape[:2]
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_area = actual_width * actual_height
        fps_error = abs(requested_fps - actual_fps) if actual_fps > 0 else requested_fps

        print(
            "Tested mode: "
            f"requested {width}x{height}@{requested_fps}, "
            f"got {actual_width}x{actual_height}@{actual_fps:.1f}"
        )

        score = (fps_error, -actual_fps, -actual_area)

        if best_score is None or score < best_score:
            best_score = score
            best_mode = (actual_width, actual_height, actual_fps)

        if actual_fps >= requested_fps and actual_width == width and actual_height == height:
            best_mode = (actual_width, actual_height, actual_fps)
            break

    if best_mode is not None:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_mode[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_mode[1])
        cap.set(cv2.CAP_PROP_FPS, requested_fps)


def detect_objects(
    model: YOLO,
    frame: Any,
    target_class: str,
    target_class_id: int | None,
    min_confidence: float,
    screen_centre: tuple[int, int],
    inference_size: int,
    device: str | None,
) -> list[dict[str, Any]]:
    """Build all valid target candidates for the current frame."""
    class_filter = [target_class_id] if target_class_id is not None else None
    model_options = {
        "conf": min_confidence,
        "imgsz": inference_size,
        "classes": class_filter,
        "verbose": False,
    }

    if device is not None:
        model_options["device"] = device

    results = model(frame, **model_options)
    candidates: list[dict[str, Any]] = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names.get(class_id, str(class_id))
            confidence = float(box.conf[0])

            if label != target_class or confidence < min_confidence:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            centre_x = int((x1 + x2) / 2)
            centre_y = int((y1 + y2) / 2)
            error_x = centre_x - screen_centre[0]
            error_y = centre_y - screen_centre[1]
            distance_squared = error_x * error_x + error_y * error_y
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

            candidates.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                    "centre": (centre_x, centre_y),
                    "distance_squared": distance_squared,
                    "area": area,
                }
            )

    return candidates


def choose_best_target(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose by centre distance, using confidence and size for near ties."""
    if not candidates:
        return None

    nearest_distance = min(
        candidate["distance_squared"] ** 0.5 for candidate in candidates
    )
    similar_candidates = [
        candidate
        for candidate in candidates
        if candidate["distance_squared"] ** 0.5
        <= nearest_distance + SIMILAR_DISTANCE_PIXELS
    ]

    # When positions are effectively tied, prefer the more reliable detection.
    return max(
        similar_candidates,
        key=lambda candidate: (candidate["confidence"], candidate["area"]),
    )


def distance_squared(
    first_centre: tuple[int, int],
    second_centre: tuple[int, int],
) -> int:
    """Return squared pixel distance between two centre points."""
    error_x = first_centre[0] - second_centre[0]
    error_y = first_centre[1] - second_centre[1]
    return error_x * error_x + error_y * error_y


def smooth_target_centre(
    new_centre: tuple[int, int],
    previous_centre: tuple[int, int] | None,
    smoothing_alpha: float,
) -> tuple[int, int]:
    """Apply exponential smoothing to the target centre."""
    smoothing_alpha = max(0.0, min(1.0, smoothing_alpha))

    if previous_centre is None:
        return new_centre

    smoothed_x = smoothing_alpha * new_centre[0] + (1.0 - smoothing_alpha) * previous_centre[0]
    smoothed_y = smoothing_alpha * new_centre[1] + (1.0 - smoothing_alpha) * previous_centre[1]
    return (round(smoothed_x), round(smoothed_y))


def copy_tracked_detection(
    detection: dict[str, Any],
    centre: tuple[int, int],
    state: str,
    visible: bool,
) -> dict[str, Any]:
    """Return a detection copy annotated with lock information."""
    tracked_detection = detection.copy()
    tracked_detection["raw_centre"] = detection["centre"]
    tracked_detection["centre"] = centre
    tracked_detection["lock_state"] = state
    tracked_detection["visible"] = visible
    return tracked_detection


def reset_target_lock(target_lock: dict[str, Any]) -> None:
    """Clear the remembered target lock state."""
    target_lock["state"] = NO_TARGET
    target_lock["last_detection"] = None
    target_lock["last_raw_centre"] = None
    target_lock["smoothed_centre"] = None
    target_lock["lost_since"] = None


def lock_match_distance_limit(detection: dict[str, Any] | None) -> float:
    """Return the largest reasonable centre jump for the current locked target."""
    if detection is None:
        return float(MIN_LOCK_MATCH_DISTANCE)

    x1, y1, x2, y2 = detection["box"]
    width = x2 - x1
    height = y2 - y1
    return max(float(MIN_LOCK_MATCH_DISTANCE), width, height)


def mark_target_lost(
    target_lock: dict[str, Any],
    current_time: float,
) -> dict[str, Any] | None:
    """Keep the last target position briefly after the target disappears."""
    if target_lock["lost_since"] is None:
        target_lock["lost_since"] = current_time

    if current_time - target_lock["lost_since"] <= LOST_TIMEOUT:
        target_lock["state"] = LOST
        last_detection = target_lock["last_detection"]
        if last_detection is None:
            return None

        lost_detection = last_detection.copy()
        lost_detection["lock_state"] = LOST
        lost_detection["visible"] = False
        return lost_detection

    reset_target_lock(target_lock)
    return None


def update_target_lock(
    candidates: list[dict[str, Any]],
    target_lock: dict[str, Any],
    current_time: float,
    smoothing_alpha: float,
) -> dict[str, Any] | None:
    """
    Keep a stable target lock across frames.

    A fresh lock uses the normal target scoring. Once locked, the tracker follows
    the closest new detection to the previous target centre instead of switching
    to a different candidate each frame. If the target vanishes briefly, the last
    smoothed position is kept until LOST_TIMEOUT expires.
    """
    if (
        target_lock["state"] == LOST
        and target_lock["lost_since"] is not None
        and current_time - target_lock["lost_since"] > LOST_TIMEOUT
    ):
        reset_target_lock(target_lock)

    previous_raw_centre = target_lock["last_raw_centre"]
    previous_smoothed_centre = target_lock["smoothed_centre"]
    has_active_lock = target_lock["state"] in (LOCKED, LOST) and previous_raw_centre is not None

    if has_active_lock and candidates:
        matched_detection = min(
            candidates,
            key=lambda candidate: distance_squared(candidate["centre"], previous_raw_centre),
        )
        match_distance_limit = lock_match_distance_limit(target_lock["last_detection"])

        if (
            distance_squared(matched_detection["centre"], previous_raw_centre)
            > match_distance_limit * match_distance_limit
        ):
            return mark_target_lost(target_lock, current_time)

        smoothed_centre = smooth_target_centre(
            matched_detection["centre"],
            previous_smoothed_centre,
            smoothing_alpha,
        )
        tracked_detection = copy_tracked_detection(
            matched_detection,
            smoothed_centre,
            LOCKED,
            True,
        )

        target_lock["state"] = LOCKED
        target_lock["last_detection"] = tracked_detection
        target_lock["last_raw_centre"] = matched_detection["centre"]
        target_lock["smoothed_centre"] = smoothed_centre
        target_lock["lost_since"] = None
        return tracked_detection

    if has_active_lock and not candidates:
        return mark_target_lost(target_lock, current_time)

    selected_detection = choose_best_target(candidates)
    if selected_detection is None:
        reset_target_lock(target_lock)
        return None

    smoothed_centre = smooth_target_centre(
        selected_detection["centre"],
        previous_smoothed_centre,
        smoothing_alpha,
    )
    tracked_detection = copy_tracked_detection(
        selected_detection,
        smoothed_centre,
        LOCKED,
        True,
    )

    target_lock["state"] = LOCKED
    target_lock["last_detection"] = tracked_detection
    target_lock["last_raw_centre"] = selected_detection["centre"]
    target_lock["smoothed_centre"] = smoothed_centre
    target_lock["lost_since"] = None
    return tracked_detection


def find_class_id(model: YOLO, target_class: str) -> int | None:
    """Find the numeric YOLO class id for a class name such as person."""
    for class_id, class_name in model.names.items():
        if class_name == target_class:
            return int(class_id)

    return None


def optimize_model(model: YOLO) -> None:
    """Apply safe Ultralytics optimizations when available."""
    try:
        model.fuse()
    except (AttributeError, RuntimeError):
        pass


def calculate_mouse_command(
    desktop_target: tuple[int, int] | None,
    cursor_position: tuple[int, int],
    deadzone: int,
    sensitivity: float,
    max_speed: int,
    tracking_state: str,
) -> dict[str, Any]:
    """Move toward the desktop target until the cursor reaches its dead zone."""
    if desktop_target is None or tracking_state != LOCKED:
        return {
            "error": (0, 0),
            "move": (0, 0),
            "command": "STOP",
            "status": tracking_state,
        }

    error_x = desktop_target[0] - cursor_position[0]
    error_y = desktop_target[1] - cursor_position[1]

    # Move the pointer toward the detected target's position in the frame.
    movement_x = 0 if abs(error_x) <= deadzone else error_x * sensitivity
    movement_y = 0 if abs(error_y) <= deadzone else error_y * sensitivity

    move_x = clamp(round(movement_x), -max_speed, max_speed)
    move_y = clamp(round(movement_y), -max_speed, max_speed)

    command = "STOP" if move_x == 0 and move_y == 0 else f"MOVE {move_x} {move_y}"

    return {
        "error": (error_x, error_y),
        "move": (move_x, move_y),
        "command": command,
        "status": tracking_state,
    }


def draw_hud_line(
    frame: Any,
    label: str,
    value: str,
    y_position: int,
) -> None:
    """Draw one readable HUD row with a shadow."""
    text = f"{label}: {value}"
    origin = (15, y_position)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, text, origin, font, 0.65, (0, 0, 0), 4)
    cv2.putText(frame, text, origin, font, 0.65, (255, 255, 255), 2)


def draw_overlay(
    frame: Any,
    candidates: list[dict[str, Any]],
    detection: dict[str, Any] | None,
    movement: dict[str, Any],
    processing_fps: float,
    camera_fps: float,
    inference_ms: float,
    target_class: str,
    serial_status: str,
    desktop_target: tuple[int, int] | None,
    cursor_position: tuple[int, int],
) -> None:
    """Draw the target overlay, crosshair, guide line, and HUD."""
    height, width = frame.shape[:2]
    screen_centre = (width // 2, height // 2)
    smoothed_centre = detection["centre"] if detection else None
    raw_centre = detection["raw_centre"] if detection else None
    confidence = detection["confidence"] if detection else 0.0
    lock_state = detection["lock_state"] if detection else NO_TARGET
    visible = detection.get("visible", False) if detection else False

    # Thin boxes show every detection that passed class/confidence filtering.
    for candidate in candidates:
        x1, y1, x2, y2 = candidate["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)

    cv2.drawMarker(
        frame,
        screen_centre,
        (255, 0, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=32,
        thickness=2,
    )

    if detection is not None:
        x1, y1, x2, y2 = detection["box"]
        box_colour = (0, 255, 0) if visible else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, 2)
        if visible:
            cv2.circle(frame, raw_centre, 6, (255, 0, 255), -1)
        cv2.circle(frame, smoothed_centre, 6, (0, 0, 255), -1)
        cv2.line(frame, screen_centre, smoothed_centre, (0, 255, 255), 2)

        cv2.putText(
            frame,
            f"{detection['label']} {confidence:.2f}",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            box_colour,
            2,
        )

    raw_error = (
        (raw_centre[0] - screen_centre[0], raw_centre[1] - screen_centre[1])
        if raw_centre and visible
        else (0, 0)
    )
    hud_rows = [
        ("Camera FPS", f"{camera_fps:.1f}"),
        ("Processing FPS", f"{processing_fps:.1f}"),
        ("Inference time", f"{inference_ms:.1f} ms"),
        ("Target class", target_class),
        ("Target lock", lock_state),
        ("Confidence", f"{confidence:.2f}"),
        ("Raw target centre", str(raw_centre) if raw_centre else "None"),
        ("Smoothed centre", str(smoothed_centre) if smoothed_centre else "None"),
        ("Frame centre", str(screen_centre)),
        ("Desktop target", str(desktop_target) if desktop_target else "None"),
        ("Cursor position", str(cursor_position)),
        ("Raw error", str(raw_error)),
        ("Desktop error", str(movement["error"])),
        ("Status", movement["status"]),
        ("Movement command", movement["command"]),
        ("Arduino", serial_status),
    ]

    for row_index, (label, value) in enumerate(hud_rows):
        draw_hud_line(frame, label, value, 30 + row_index * 28)


def main() -> None:
    args = parse_args()
    if args.list_serial_ports:
        print_serial_ports()
        return

    model = YOLO(args.model)
    optimize_model(model)
    target_class_id = find_class_id(model, args.target_class)
    cap = open_camera(args.source, args.width, args.height, args.fps)
    try:
        arduino = ArduinoController(args.serial_port, args.serial_baud) if args.serial_port else None
    except (serial.SerialException, OSError):
        cap.release()
        raise

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print("Vision HID Controller started")
    print(f"Target class: {args.target_class}")
    if target_class_id is not None:
        print(f"Target class id: {target_class_id}")
    else:
        print("Target class id: not found, filtering by class name after detection")
    print(f"Minimum confidence: {args.confidence:.2f}")
    print(f"Capture size: {actual_width}x{actual_height}")
    print(f"Requested capture FPS: {args.fps}")
    print(f"Reported capture FPS: {actual_fps:.1f}")
    print(f"YOLO inference size: {args.imgsz}")
    print(f"Arduino: {args.serial_port if arduino else 'simulation only'}")
    print("Press F8 anywhere to pause/resume tracking output")
    print("Press q or Esc in the video window to quit")

    previous_time = time.perf_counter()
    last_command = ""
    target_lock: dict[str, Any] = {
        "state": NO_TARGET,
        "last_detection": None,
        "last_raw_centre": None,
        "smoothed_centre": None,
        "lost_since": None,
    }
    pause_hotkey = GlobalPauseHotkey()
    pause_hotkey.start()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("STOP")
                break

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time
            processing_fps = 1.0 / elapsed if elapsed > 0 else 0.0

            height, width = frame.shape[:2]
            screen_centre = (width // 2, height // 2)
            inference_start = time.perf_counter()
            candidates = detect_objects(
                model,
                frame,
                args.target_class,
                target_class_id,
                args.confidence,
                screen_centre,
                args.imgsz,
                args.device,
            )
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            detection = update_target_lock(
                candidates,
                target_lock,
                current_time,
                args.smoothing,
            )
            target_centre = detection["centre"] if detection else None
            desktop_target = (
                map_frame_point_to_desktop(target_centre, (width, height))
                if target_centre is not None
                else None
            )
            cursor_position = get_desktop_cursor_position()
            movement = calculate_mouse_command(
                desktop_target,
                cursor_position,
                args.deadzone,
                args.sensitivity,
                args.max_speed,
                detection["lock_state"] if detection else NO_TARGET,
            )

            if pause_hotkey.paused.is_set():
                movement = {
                    "error": movement["error"],
                    "move": (0, 0),
                    "command": "STOP",
                    "status": "PAUSED",
                }

            if movement["command"] != last_command:
                print(movement["command"])
                last_command = movement["command"]

            if arduino is not None:
                arduino.send_command(movement["command"])

            draw_overlay(
                frame,
                candidates,
                detection,
                movement,
                processing_fps,
                actual_fps,
                inference_ms,
                args.target_class,
                (
                    "PAUSED"
                    if pause_hotkey.paused.is_set()
                    else "ARMED" if arduino else "SIMULATION"
                ),
                desktop_target,
                cursor_position,
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            window_visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)

            if key in (ord("q"), 27) or window_visible < 1:
                print("STOP")
                break
    finally:
        pause_hotkey.stop()
        if arduino is not None:
            arduino.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
