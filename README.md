# Vision HID Controller

Vision HID Controller is a Python + Arduino starter project for turning computer
vision detections into mouse-style HID commands.

The Python program always prints commands such as `MOVE 12 -5` and `STOP` for
debugging. With `--serial-port`, it also sends them to an Arduino-compatible
USB HID board to control the mouse.

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

Press `F8` from any Windows application to pause or resume mouse movement. This
global hotkey does not require the OpenCV window to be selected. Detection and
the HUD continue running while movement output remains at `STOP`.

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
- Draws valid candidate boxes and highlights the locked target.
- Draws a HUD with FPS, inference time, tracking state, confidence, raw and
  smoothed target centres, frame centre, errors, and the movement command.
- Smooths the target position, then prints scaled and clamped commands such as
  `MOVE 12 -5`.
- Prints `STOP` when no target is locked, a target is temporarily lost, or the
  smoothed target position is inside the dead zone around the crosshair.

## Target Selection Algorithm

For each YOLO detection, the script checks that the class is `person` and the
confidence is at least `0.60`. It then calculates the detection centre and
compares it to the screen centre:

```text
dx = target_x - screen_centre_x
dy = target_y - screen_centre_y
distance_squared = dx * dx + dy * dy
```

The target with the shortest Euclidean distance is selected. Targets within 10
pixels of the nearest distance are treated as similar: higher confidence wins,
then larger bounding-box area.

## Dead Zone Behavior

The dead zone prevents jitter when the desktop cursor is already close to the
mapped target position. If the cursor is within the dead zone, the HUD shows
`LOCKED` and the command is `STOP`. `STOP` also applies when no valid target is
locked, the lock is temporarily lost, or the script is exiting.

The smoothed target point is mapped proportionally from frame coordinates onto
the primary desktop. The controller compares that desktop point with the live
Windows cursor position, moves by the remaining error, and stops when the cursor
reaches the dead zone around the target. This closed loop prevents continuous
movement after the cursor reaches the item.

## Target Smoothing

The detected target position uses an exponential low-pass filter:

```text
smoothed = smoothing * raw + (1 - smoothing) * previous_smoothed
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

## Arduino HID Setup

The Arduino sketch is included for later hardware integration. Use an Arduino
Leonardo, Micro, Pro Micro, or another ATmega32U4-based board.

This requires a native USB board supported by Arduino's `Mouse` library, such as
an Arduino Leonardo, Micro, or ATmega32U4-based Pro Micro. A regular Uno cannot
use this sketch as a native USB mouse.

Open this sketch in the Arduino IDE:

```text
arduino/Vision_HID_Controller/Vision_HID_Controller.ino
```

Select the correct board and port, then upload it. The board starts **disarmed**
and cannot move the mouse until Python sends `ARM`.

List available ports:

```bash
python python/vision_controller.py --list-serial-ports
```

Run with the Arduino connected (replace `COM6` and the camera index as needed):

```bash
python python/vision_controller.py --source 0 --serial-port COM6
```

The default Arduino port is `COM6`, so the IDE Run button or the following
command connects to COM6 automatically:

```bash
python python/vision_controller.py
```

To run detection without Arduino mouse movement:

```bash
python python/vision_controller.py --no-arduino
```

The serial protocol is deliberately small:

- `ARM` enables HID movement.
- `MOVE dx dy` performs one relative mouse movement.
- `STOP` releases mouse buttons but keeps the connection armed.
- `DISARM` blocks movement until another `ARM`.

Python repeats the current command as a heartbeat. If messages stop for 5 seconds,
the Arduino watchdog automatically disarms HID. Python periodically refreshes
`ARM`, allowing control to recover after a temporary camera or inference stall.
Closing the Python window sends `STOP` and `DISARM` before closing the serial port.

## Safety Notes

The Arduino sketch moves the real system pointer. Keep `MAX_SPEED` small during
testing and keep the board's USB cable within reach. Press `q` or `Esc` in the
OpenCV window to stop; unplugging the board is the final emergency stop.
