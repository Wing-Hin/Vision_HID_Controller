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
CLICK_CONFIDENCE = 0.90
CLICK_COOLDOWN_SECONDS = 1.0

REQUESTED_WIDTH = 1280
REQUESTED_HEIGHT = 720
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
        cap = cv2.VideoCapture(index)

        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                height, width = frame.shape[:2]
                cameras.append({"index": index, "width": width, "height": height})

        cap.release()

    return cameras


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


def open_camera(source: str | None) -> cv2.VideoCapture:
    """
    Open a camera, video file, or stream.

    Numeric sources use OpenCV camera indexes. When source is None, available
    cameras are printed and the user chooses one from the console.
    """
    if source is None:
        cameras = discover_cameras()
        camera_index = choose_camera(cameras)
        cap = cv2.VideoCapture(camera_index)
    elif source.isdigit():
        camera_index = int(source)
        remember_camera(camera_index)
        cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    configure_capture_resolution(cap)
    return cap


def configure_capture_resolution(cap: cv2.VideoCapture) -> None:
    """
    Request 1280x720, then fall back to the highest practical resolution found.

    Many cameras quietly choose the nearest supported mode. We test a short list
    of common sizes and keep the largest area that returns a valid frame.
    """
    candidates = [
        (REQUESTED_WIDTH, REQUESTED_HEIGHT),
        (3840, 2160),
        (2560, 1440),
        (1920, 1080),
        (1600, 900),
        (1280, 720),
        (1024, 768),
        (800, 600),
        (640, 480),
    ]
    best_size = None
    best_area = 0

    for width, height in candidates:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        ok, frame = cap.read()
        if not ok:
            continue

        actual_height, actual_width = frame.shape[:2]
        actual_area = actual_width * actual_height

        if (actual_width, actual_height) == (REQUESTED_WIDTH, REQUESTED_HEIGHT):
            best_size = (actual_width, actual_height)
            break

        if actual_area > best_area:
            best_area = actual_area
            best_size = (actual_width, actual_height)

    if best_size is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_size[1])


def detect_objects(
    model: YOLO,
    frame: Any,
    target_class: str,
    min_confidence: float,
    screen_centre: tuple[int, int],
) -> dict[str, Any] | None:
    """Find the target-class detection closest to the screen centre."""
    results = model(frame, conf=min_confidence, verbose=False)
    closest_detection = None
    closest_distance_squared = None

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

            if (
                closest_distance_squared is None
                or distance_squared < closest_distance_squared
            ):
                closest_distance_squared = distance_squared
                closest_detection = {
                    "label": label,
                    "confidence": confidence,
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                    "centre": (centre_x, centre_y),
                    "distance_squared": distance_squared,
                }

    return closest_detection


def calculate_mouse_command(
    target_centre: tuple[int, int] | None,
    screen_centre: tuple[int, int],
    deadzone: int,
    sensitivity: float,
    max_speed: int,
) -> dict[str, Any]:
    """Convert raw pixel error into a smooth, clamped MOVE command."""
    if target_centre is None:
        return {
            "raw_error": (0, 0),
            "move": (0, 0),
            "command": "STOP",
            "status": "NO TARGET",
        }

    raw_dx = target_centre[0] - screen_centre[0]
    raw_dy = target_centre[1] - screen_centre[1]

    move_x = 0 if abs(raw_dx) <= deadzone else int(raw_dx * sensitivity)
    move_y = 0 if abs(raw_dy) <= deadzone else int(raw_dy * sensitivity)

    move_x = clamp(move_x, -max_speed, max_speed)
    move_y = clamp(move_y, -max_speed, max_speed)

    target_is_centered = move_x == 0 and move_y == 0
    command = "MOVE 0 0" if target_is_centered else f"MOVE {move_x} {move_y}"
    status = "LOCKED" if target_is_centered else "TRACKING"

    return {
        "raw_error": (raw_dx, raw_dy),
        "move": (move_x, move_y),
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
    fps: float,
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
        ("FPS", f"{fps:.1f}"),
        ("Target class", target_class),
        ("Confidence", f"{confidence:.2f}"),
        ("Target centre", target_text),
        ("Screen centre", str(screen_centre)),
        ("Target distance", f"{target_distance:.1f}px"),
        ("Raw error", str(movement["raw_error"])),
        ("Status", movement["status"]),
        ("Movement command", movement["command"]),
    ]

    for row_index, (label, value) in enumerate(hud_rows):
        draw_hud_line(frame, label, value, 30 + row_index * 28)


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    cap = open_camera(args.source)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Vision HID Controller started")
    print(f"Target class: {args.target_class}")
    print(f"Minimum confidence: {args.confidence:.2f}")
    print(f"Capture size: {actual_width}x{actual_height}")
    print("Press q or Esc in the video window to quit")

    previous_time = time.perf_counter()
    last_click_time = 0.0
    last_command = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            print("STOP")
            break

        current_time = time.perf_counter()
        elapsed = current_time - previous_time
        previous_time = current_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        height, width = frame.shape[:2]
        screen_centre = (width // 2, height // 2)
        detection = detect_objects(
            model,
            frame,
            args.target_class,
            args.confidence,
            screen_centre,
        )
        target_centre = detection["centre"] if detection else None
        movement = calculate_mouse_command(
            target_centre,
            screen_centre,
            args.deadzone,
            args.sensitivity,
            args.max_speed,
        )

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

        draw_overlay(frame, detection, movement, fps, args.target_class)
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
