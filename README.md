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
- Requests `60 FPS` capture from the selected camera.
- Uses MJPG capture mode when available, which helps many webcams reach 60 FPS.
- If 1280x720 is capped at 30 FPS, tries smaller common modes such as 960x540
  640x360, 640x480, 424x240, and 320x240 to find a 60 FPS mode.
- Creates a large resizable OpenCV window.
- Tracks only the configured target class, currently `person`.
- Ignores detections below `0.60` confidence.
- If multiple people are visible, chooses the person whose bounding-box centre is
  closest to the blue screen-centre crosshair.
- Draws a HUD with FPS, confidence, target centre, screen centre, raw error, and
  movement command.
- Prints smoothed and clamped movement commands such as `MOVE 12 -5`.
- Shows `LOCKED` and prints `MOVE 0 0` when the target is inside the dead zone
  around the crosshair.
- Uses a low-pass filter so movement eases toward each new command instead of
  snapping instantly.

## Target Selection Algorithm

For each YOLO detection, the script checks that the class is `person` and the
confidence is at least `0.60`. It then calculates the detection centre and
compares it to the screen centre:

```text
dx = target_x - screen_centre_x
dy = target_y - screen_centre_y
distance_squared = dx * dx + dy * dy
```

The target with the smallest `distance_squared` is selected. This avoids tracking
a random person when multiple people are visible and avoids the extra square
root needed for true Euclidean distance.

## Dead Zone Behavior

The dead zone prevents jitter when the selected target is already close to the
crosshair. If the target is detected but close enough to the centre, the HUD
shows `LOCKED` and the printed command is `MOVE 0 0`. `STOP` means no valid
target is currently detected or the script is exiting.

## Movement Smoothing

Movement uses a low-pass filter:

```text
filtered = previous + smoothing * (target - previous)
```

The default smoothing value is `0.25`. Lower values feel smoother but respond
more slowly. Higher values react faster but can look more jumpy.

You can tune it from the command line:

```bash
python python/vision_controller.py --smoothing 0.15
```

## Capture FPS

The script requests 60 FPS by default:

```bash
python python/vision_controller.py
```

You can request another frame rate:

```bash
python python/vision_controller.py --fps 30
```

You can also request a smaller starting resolution:

```bash
python python/vision_controller.py --width 640 --height 480 --fps 60
```

Some webcams and virtual cameras ignore FPS requests or only support 60 FPS at
specific resolutions. Check the startup line `Reported capture FPS` to see what
OpenCV reports after opening the camera.

If `Reported capture FPS` is 60 but the HUD FPS is around 30, YOLO inference is
the bottleneck. Try a smaller inference size:

```bash
python python/vision_controller.py --imgsz 320
```

The default inference size is already tuned for speed at `320`. You can go lower
for more FPS, with less detection detail:

```bash
python python/vision_controller.py --imgsz 256
```

If PyTorch can see a supported GPU, you can try forcing it:

```bash
python python/vision_controller.py --device 0
```

The HUD shows two FPS values:

- `Camera FPS`: the capture rate reported by OpenCV for the selected camera.
- `Processing FPS`: the full loop speed after camera read, YOLO inference,
  overlay drawing, and display.

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
