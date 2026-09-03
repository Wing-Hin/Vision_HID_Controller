/*
  Vision HID Controller - Arduino Leonardo / Micro / Pro Micro (ATmega32U4).

  Serial protocol at 115200 baud:
    ARM          allow MOVE commands
    MOVE 10 -5   move the USB mouse relatively
    STOP         release mouse buttons; remain armed
    DISARM       ignore MOVE commands until the next ARM

  Safety behavior:
  - The board always starts disarmed.
  - Invalid or out-of-range commands do not move the mouse.
  - If Python stops sending commands for 5 seconds, the board disarms itself.
*/

#include <Mouse.h>

const unsigned long SERIAL_BAUD = 115200;
const int MAX_MOVE = 50;
const unsigned long COMMAND_TIMEOUT_MS = 5000;
const byte COMMAND_BUFFER_SIZE = 32;

char commandBuffer[COMMAND_BUFFER_SIZE];
byte commandLength = 0;
bool discardingCommand = false;
bool hidArmed = false;
unsigned long lastCommandTime = 0;

void setup() {
  Serial.begin(SERIAL_BAUD);
  Mouse.begin();
  releaseMouseButtons();
}

void loop() {
  readSerialCommands();

  if (hidArmed && millis() - lastCommandTime > COMMAND_TIMEOUT_MS) {
    disarmHid();
    Serial.println("TIMEOUT DISARMED");
  }
}

void readSerialCommands() {
  while (Serial.available() > 0) {
    char incoming = Serial.read();

    if (incoming == '\r') {
      continue;
    }

    if (incoming == '\n') {
      if (discardingCommand) {
        discardingCommand = false;
        commandLength = 0;
        continue;
      }

      commandBuffer[commandLength] = '\0';
      if (commandLength > 0) {
        handleCommand(commandBuffer);
      }
      commandLength = 0;
      continue;
    }

    if (discardingCommand) {
      continue;
    }

    if (commandLength < COMMAND_BUFFER_SIZE - 1) {
      commandBuffer[commandLength] = incoming;
      commandLength++;
    } else {
      // Drop an oversized line instead of executing a truncated command.
      commandLength = 0;
      discardingCommand = true;
      Serial.println("ERR command too long");
    }
  }
}

void handleCommand(const char *command) {
  if (strcmp(command, "ARM") == 0) {
    hidArmed = true;
    lastCommandTime = millis();
    Serial.println("OK ARMED");
    return;
  }

  if (strcmp(command, "STOP") == 0) {
    releaseMouseButtons();
    lastCommandTime = millis();
    Serial.println("OK STOP");
    return;
  }

  if (strcmp(command, "DISARM") == 0) {
    disarmHid();
    Serial.println("OK DISARMED");
    return;
  }

  if (strncmp(command, "MOVE ", 5) == 0) {
    handleMove(command + 5);
    return;
  }

  Serial.println("ERR unknown command");
}

void handleMove(const char *arguments) {
  char *afterDx;
  long dx = strtol(arguments, &afterDx, 10);

  if (afterDx == arguments || *afterDx != ' ') {
    Serial.println("ERR MOVE requires integer dx and dy");
    return;
  }

  const char *dyText = afterDx + 1;
  char *afterDy;
  long dy = strtol(dyText, &afterDy, 10);

  if (afterDy == dyText || *afterDy != '\0') {
    Serial.println("ERR MOVE requires integer dx and dy");
    return;
  }

  if (dx < -MAX_MOVE || dx > MAX_MOVE || dy < -MAX_MOVE || dy > MAX_MOVE) {
    Serial.println("ERR MOVE outside safety limit");
    return;
  }

  if (!hidArmed) {
    Serial.println("ERR DISARMED");
    return;
  }

  lastCommandTime = millis();
  Mouse.move((signed char)dx, (signed char)dy, 0);
  Serial.println("OK MOVE");
}

void releaseMouseButtons() {
  Mouse.release(MOUSE_LEFT);
  Mouse.release(MOUSE_RIGHT);
  Mouse.release(MOUSE_MIDDLE);
}

void disarmHid() {
  releaseMouseButtons();
  hidArmed = false;
}
