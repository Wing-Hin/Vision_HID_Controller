# Vision HID Controller

Vision HID Controller is a Python + Arduino starter project for turning computer
vision detections into mouse-style HID commands.

This version focuses on the Python vision system only. It prints simulated
commands such as `MOVE 12 -5`, `CLICK`, and `STOP`; it does not send commands to
the Arduino yet.

## Project Structure

```text
Vision_HID_Controller/
+-- python/
|   +-- vision_controller.py
+-- arduino/
|   +-- Vision_HID_Controller/
|       +-- Vision_HID_Controller.ino
+-- docs/
+-- requirements.txt
+-- README.md
```

## Python Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Run the vision controller:

```bash
python python/vision_controller.py
```

On startup, the script scans available cameras, prints them, and asks which one
to use. The selected camera is remembered for the next run.

You can still provide a source directly:

```bash
python python/vision_controller.py --source 1
```

Press `q`, `Esc`, or close the OpenCV window to stop.

## Vision Features

- Requests `1280x720` capture from the selected camera.
- Falls back to the highest practical resolution OpenCV can read.
- Creates a large resizable OpenCV window.
- Tracks only the configured target class, currently `person`.
- Ignores detections below `0.60` confidence.
- Draws a HUD with FPS, confidence, target centre, screen centre, raw error, and
  movement command.
- Prints smoothed and clamped movement commands such as `MOVE 12 -5`.

## Arduino Setup

The Arduino sketch is included for later hardware integration. Use an Arduino
Leonardo, Micro, Pro Micro, or another ATmega32U4-based board.

Open this sketch in the Arduino IDE:

```text
arduino/Vision_HID_Controller/Vision_HID_Controller.ino
```

This project does not connect Python to Arduino serial yet.

## Safety Notes

The Python script only prints simulated commands. The Arduino sketch can move and
click the real mouse once serial integration is added later, so keep movement
limits small during future hardware testing.
