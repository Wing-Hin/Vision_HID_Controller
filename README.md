# Vision HID Controller

Vision HID Controller is a simple Python + Arduino starter project for turning
object detections into mouse-style HID commands.

The first Python version is intentionally safe: it prints simulated commands
such as `MOVE 10 -5`, `CLICK`, and `STOP` instead of sending them directly to the
Arduino. The Arduino sketch is ready to receive those commands over serial and
use the `Mouse` library on an Arduino Leonardo / ATmega32U4-compatible board.

## Project Structure

```text
Vision_HID_Controller/
├── python/
│   └── vision_controller.py
├── arduino/
│   └── Vision_HID_Controller/
│       └── Vision_HID_Controller.ino
├── docs/
├── requirements.txt
└── README.md
```

## Python Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Run with the default camera source:

```bash
python python/vision_controller.py
```

The script defaults to camera source `1`. Use source `0` if your webcam is the
first camera device:

```bash
python python/vision_controller.py --source 0
```

Press `q` in the video window to stop.

## What The Python Script Does

- Opens a webcam, OBS Virtual Camera, or video file.
- Runs an Ultralytics YOLO model on each frame.
- Draws a bounding box around the highest-confidence detected object.
- Calculates the centre point of that object.
- Compares the object centre with the frame centre.
- Prints simulated commands:
  - `MOVE dx dy` when the target is away from the frame centre.
  - `CLICK` when confidence is high enough.
  - `STOP` when the script exits.

## Arduino Setup

Use an Arduino Leonardo, Micro, Pro Micro, or another ATmega32U4-based board.
Those boards can act as USB HID mouse devices.

1. Open `arduino/Vision_HID_Controller/Vision_HID_Controller.ino` in the Arduino
   IDE.
2. Select your Leonardo / ATmega32U4 board and port.
3. Upload the sketch.
4. Open the Serial Monitor at `115200` baud to test commands.

Example serial commands:

```text
MOVE 10 -5
CLICK
STOP
START
```

## Safety Notes

The Arduino sketch can move and click the real mouse. Keep movements small while
testing, and use `STOP` to disable HID actions. Send `START` to enable them
again.
