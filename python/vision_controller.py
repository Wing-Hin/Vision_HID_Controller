"""
Vision HID Controller - Python vision loop.

This first version reads frames from a webcam or OBS Virtual Camera, runs a YOLO
model, draws boxes, finds the highest-confidence detection, and prints simulated
HID commands such as "MOVE dx dy" and "CLICK".
"""

from __future__ import annotations

import argparse
import time

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision HID Controller")
    parser.add_argument(
        "--source",
        default="1",
        help="Camera index such as 1, or a video path. Use 0 if your webcam is the first device.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics YOLO model path or name. yolov8n.pt is small and quick.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum detection confidence.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Requested camera capture width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Requested camera capture height.",
    )
    parser.add_argument(
        "--deadzone",
        type=int,
        default=25,
        help="Ignore small movements around the frame centre.",
    )
    parser.add_argument(
        "--move-scale",
        type=float,
        default=0.05,
        help="Scale pixel offset into simulated mouse movement.",
    )
    parser.add_argument(
        "--click-confidence",
        type=float,
        default=0.80,
        help="Print CLICK when the best detection is above this confidence.",
    )
    parser.add_argument(
        "--click-cooldown",
        type=float,
        default=1.0,
        help="Minimum seconds between simulated CLICK commands.",
    )
    return parser.parse_args()


def open_video_source(source: str) -> cv2.VideoCapture:
    """Open a numeric camera index or a video file/stream string."""
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def set_capture_size(cap: cv2.VideoCapture, width: int, height: int) -> None:
    """Ask the camera for a capture size. The device may choose a fallback."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def best_detection(results, min_confidence: float):
    """Return the highest-confidence YOLO box, or None when no box qualifies."""
    best_box = None
    best_confidence = min_confidence

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence > best_confidence:
                best_confidence = confidence
                best_box = box

    return best_box


def movement_from_target(
    target_x: int,
    target_y: int,
    frame_width: int,
    frame_height: int,
    deadzone: int,
    scale: float,
) -> tuple[int, int]:
    """Convert target position into small mouse-like movement deltas."""
    offset_x = target_x - frame_width // 2
    offset_y = target_y - frame_height // 2

    if abs(offset_x) < deadzone:
        offset_x = 0
    if abs(offset_y) < deadzone:
        offset_y = 0

    return int(offset_x * scale), int(offset_y * scale)


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    cap = open_video_source(args.source)
    set_capture_size(cap, args.width, args.height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    last_click_time = 0.0

    print("Vision HID Controller started")
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Requested capture size: {args.width}x{args.height}")
    print(f"Actual capture size: {actual_width}x{actual_height}")
    print("Press q or Esc in the video window to quit")
    cv2.namedWindow("Vision HID Controller", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("STOP")
            break

        frame_height, frame_width = frame.shape[:2]
        results = model(frame, conf=args.confidence, verbose=False)
        selected_box = best_detection(results, args.confidence)

        if selected_box is not None:
            x1, y1, x2, y2 = selected_box.xyxy[0].tolist()
            confidence = float(selected_box.conf[0])
            class_id = int(selected_box.cls[0])
            label = model.names.get(class_id, str(class_id))

            centre_x = int((x1 + x2) / 2)
            centre_y = int((y1 + y2) / 2)
            move_x, move_y = movement_from_target(
                centre_x,
                centre_y,
                frame_width,
                frame_height,
                args.deadzone,
                args.move_scale,
            )

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )
            cv2.circle(frame, (centre_x, centre_y), 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (int(x1), max(20, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            if move_x != 0 or move_y != 0:
                print(f"MOVE {move_x} {move_y}")

            now = time.monotonic()
            if (
                confidence >= args.click_confidence
                and now - last_click_time >= args.click_cooldown
            ):
                print("CLICK")
                last_click_time = now

        cv2.drawMarker(
            frame,
            (frame_width // 2, frame_height // 2),
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2,
        )
        cv2.imshow("Vision HID Controller", frame)

        key = cv2.waitKey(1) & 0xFF
        window_visible = cv2.getWindowProperty(
            "Vision HID Controller",
            cv2.WND_PROP_VISIBLE,
        )

        if key in (ord("q"), 27) or window_visible < 1:
            print("STOP")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
