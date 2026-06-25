/*
  Vision HID Controller - Arduino Leonardo / ATmega32U4 sketch.

  Receives serial commands from a host computer:
    MOVE 10 -5
    CLICK
    STOP

  Boards based on the ATmega32U4 can act as USB HID devices. Examples include
  Arduino Leonardo, Micro, and Pro Micro. Upload with care because this sketch
  can control the host mouse.
*/

#include <Mouse.h>

const long SERIAL_BAUD = 115200;
const int MAX_MOVE = 50;

bool hidEnabled = true;

void setup() {
  Serial.begin(SERIAL_BAUD);
  Mouse.begin();
}

void loop() {
  if (!Serial.available()) {
    return;
  }

  String command = Serial.readStringUntil('\n');
  command.trim();

  if (command.length() == 0) {
    return;
  }

  handleCommand(command);
}

void handleCommand(String command) {
  if (command == "CLICK") {
    if (hidEnabled) {
      Mouse.click(MOUSE_LEFT);
    }
    Serial.println("OK CLICK");
    return;
  }

  if (command == "STOP") {
    hidEnabled = false;
    Serial.println("OK STOP");
    return;
  }

  if (command == "START") {
    hidEnabled = true;
    Serial.println("OK START");
    return;
  }

  if (command.startsWith("MOVE ")) {
    int firstSpace = command.indexOf(' ');
    int secondSpace = command.indexOf(' ', firstSpace + 1);

    if (secondSpace < 0) {
      Serial.println("ERR MOVE requires dx and dy");
      return;
    }

    int dx = command.substring(firstSpace + 1, secondSpace).toInt();
    int dy = command.substring(secondSpace + 1).toInt();

    dx = constrain(dx, -MAX_MOVE, MAX_MOVE);
    dy = constrain(dy, -MAX_MOVE, MAX_MOVE);

    if (hidEnabled) {
      Mouse.move(dx, dy, 0);
    }
    Serial.println("OK MOVE");
    return;
  }

  Serial.println("ERR unknown command");
}
