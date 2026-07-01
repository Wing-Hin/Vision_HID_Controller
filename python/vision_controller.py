"""
Vision HID Controller - vision-only prototype.

This version focuses on reliable camera input and object tracking. It does not
talk to the Arduino yet. It only prints simulated commands such as:

    MOVE 12 -5
    CLICK
    STOP
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


# Beginner-friendly configuration values. Tweak these first.
TARGET_CLASS = "person"
MIN_CONFIDENCE = 0.60
DEADZONE = 20
SENSITIVITY = 0.08
MAX_SPEED = 40
SMOOTHING_ALPHA = 0.25
CLICK_CONFIDENCE = 0.90
CLICK_COOLDOWN_SECONDS = 1.0

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
    return parser.parse_args()


def clamp(value: int, minimum: int, maximum: int) -> int:
    """Keep a number inside a safe range."""
    return max(minimum, min(maximum, value))


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
) -> dict[str, Any] | None:
    """Build all valid target candidates and return the best-ranked one."""
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

    if not candidates:
        return None

    # Primary: nearest to frame centre. Ties then favor confidence and size.
    return min(
        candidates,
        key=lambda candidate: (
            candidate["distance_squared"],
            -candidate["confidence"],
            -candidate["area"],
        ),
    )


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
    target_centre: tuple[int, int] | None,
    screen_centre: tuple[int, int],
    previous_move: tuple[float, float],
    deadzone: int,
    sensitivity: float,
    max_speed: int,
    smoothing_alpha: float,
) -> dict[str, Any]:
    """Convert raw pixel error into a smooth, clamped MOVE command."""
    smoothing_alpha = max(0.0, min(1.0, smoothing_alpha))

    if target_centre is None:
        return {
            "raw_error": (0, 0),
            "move": (0, 0),
            "filtered_move": (0.0, 0.0),
            "command": "STOP",
            "status": "NO TARGET",
        }

    raw_dx = target_centre[0] - screen_centre[0]
    raw_dy = target_centre[1] - screen_centre[1]

    target_move_x = 0 if abs(raw_dx) <= deadzone else raw_dx * sensitivity
    target_move_y = 0 if abs(raw_dy) <= deadzone else raw_dy * sensitivity

    target_move_x = clamp(int(target_move_x), -max_speed, max_speed)
    target_move_y = clamp(int(target_move_y), -max_speed, max_speed)

    # Low-pass filter: move partway from the previous command toward the new one.
    filtered_x = previous_move[0] + smoothing_alpha * (target_move_x - previous_move[0])
    filtered_y = previous_move[1] + smoothing_alpha * (target_move_y - previous_move[1])

    if abs(filtered_x) < 0.5:
        filtered_x = 0.0
    if abs(filtered_y) < 0.5:
        filtered_y = 0.0

    move_x = clamp(round(filtered_x), -max_speed, max_speed)
    move_y = clamp(round(filtered_y), -max_speed, max_speed)

    target_is_centered = target_move_x == 0 and target_move_y == 0
    filtered_is_stopped = move_x == 0 and move_y == 0
    command = "MOVE 0 0" if filtered_is_stopped else f"MOVE {move_x} {move_y}"

    if target_is_centered and filtered_is_stopped:
        status = "LOCKED"
    elif target_is_centered:
        status = "SETTLING"
    else:
        status = "TRACKING"

    return {
        "raw_error": (raw_dx, raw_dy),
        "move": (move_x, move_y),
        "target_move": (target_move_x, target_move_y),
        "filtered_move": (filtered_x, filtered_y),
        "command": command,
        "status": status,
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
    detection: dict[str, Any] | None,
    movement: dict[str, Any],
    processing_fps: float,
    camera_fps: float,
    target_class: str,
) -> None:
    """Draw the target overlay, crosshair, guide line, and HUD."""
    height, width = frame.shape[:2]
    screen_centre = (width // 2, height // 2)
    target_centre = detection["centre"] if detection else None
    confidence = detection["confidence"] if detection else 0.0
    target_distance = detection["distance_squared"] ** 0.5 if detection else 0.0

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
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, target_centre, 6, (0, 0, 255), -1)
        cv2.line(frame, screen_centre, target_centre, (0, 255, 255), 2)

        cv2.putText(
            frame,
            f"{detection['label']} {confidence:.2f}",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    target_text = str(target_centre) if target_centre else "None"
    hud_rows = [
        ("Camera FPS", f"{camera_fps:.1f}"),
        ("Processing FPS", f"{processing_fps:.1f}"),
        ("Target class", target_class),
        ("Confidence", f"{confidence:.2f}"),
        ("Target centre", target_text),
        ("Screen centre", str(screen_centre)),
        ("Target distance", f"{target_distance:.1f}px"),
        ("Raw error", str(movement["raw_error"])),
        ("Filtered move", f"({movement['filtered_move'][0]:.1f}, {movement['filtered_move'][1]:.1f})"),
        ("Status", movement["status"]),
        ("Movement command", movement["command"]),
    ]

    for row_index, (label, value) in enumerate(hud_rows):
        draw_hud_line(frame, label, value, 30 + row_index * 28)


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    optimize_model(model)
    target_class_id = find_class_id(model, args.target_class)
    cap = open_camera(args.source, args.width, args.height, args.fps)

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
    print("Press q or Esc in the video window to quit")

    previous_time = time.perf_counter()
    last_click_time = 0.0
    last_command = ""
    previous_move = (0.0, 0.0)

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
        detection = detect_objects(
            model,
            frame,
            args.target_class,
            target_class_id,
            args.confidence,
            screen_centre,
            args.imgsz,
            args.device,
        )
        target_centre = detection["centre"] if detection else None
        movement = calculate_mouse_command(
            target_centre,
            screen_centre,
            previous_move,
            args.deadzone,
            args.sensitivity,
            args.max_speed,
            args.smoothing,
        )
        previous_move = movement["filtered_move"]

        if movement["command"] != last_command:
            print(movement["command"])
            last_command = movement["command"]

        if (
            detection
            and detection["confidence"] >= CLICK_CONFIDENCE
            and current_time - last_click_time >= CLICK_COOLDOWN_SECONDS
        ):
            print("CLICK")
            last_click_time = current_time

        draw_overlay(
            frame,
            detection,
            movement,
            processing_fps,
            actual_fps,
            args.target_class,
        )
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        window_visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)

        if key in (ord("q"), 27) or window_visible < 1:
            print("STOP")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
