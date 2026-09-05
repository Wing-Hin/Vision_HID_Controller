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

// Communication and safety settings.
const unsigned long SERIAL_BAUD = 115200;
const int MAX_MOVE = 50;
const unsigned long COMMAND_TIMEOUT_MS = 5000;
const byte COMMAND_BUFFER_SIZE = 32;

// Temporary storage for one incoming text command.
// C strings need one extra byte at the end for the null character '\0'.
char commandBuffer[COMMAND_BUFFER_SIZE];
byte commandLength = 0;

// Becomes true when an incoming command is too long. The Arduino then ignores
// everything until the next newline instead of using an incomplete command.
bool discardingCommand = false;

// The board starts disarmed, so MOVE commands cannot move the cursor yet.
bool hidArmed = false;

// Used by the watchdog to detect when Python has stopped communicating.
unsigned long lastCommandTime = 0;

void setup() {
  // On a Leonardo/Micro, Serial is the USB virtual serial connection to Python.
  Serial.begin(SERIAL_BAUD);

  // Make the board appear to the computer as a USB HID mouse.
  Mouse.begin();

  // Make sure no mouse button is held when the board starts.
  releaseMouseButtons();
}

void loop() {
  // Read and process any commands that have arrived from Python.
  readSerialCommands();

  // Safety watchdog: if Python sends nothing for five seconds while the board
  // is armed, release all buttons and block further movement.
  if (hidArmed && millis() - lastCommandTime > COMMAND_TIMEOUT_MS) {
    disarmHid();
    Serial.println("TIMEOUT DISARMED");
  }
}

void readSerialCommands() {
  // Serial data arrives one character at a time. Keep reading until the receive
  // buffer has no more characters available.
  while (Serial.available() > 0) {
    char incoming = Serial.read();

    // Some computers end a line with "\r\n". Only "\n" is needed here.
    if (incoming == '\r') {
      continue;
    }

    // A newline marks the end of one complete command.
    if (incoming == '\n') {
      // If this line was too long, discard it and prepare for the next line.
      if (discardingCommand) {
        discardingCommand = false;
        commandLength = 0;
        continue;
      }

      // Add the null character required at the end of a C string.
      commandBuffer[commandLength] = '\0';
      if (commandLength > 0) {
        handleCommand(commandBuffer);
      }

      // Start filling the buffer from the beginning for the next command.
      commandLength = 0;
      continue;
    }

    // Ignore the rest of an oversized command until its newline arrives.
    if (discardingCommand) {
      continue;
    }

    // Leave one free byte in the array for the final null character.
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
  // strcmp returns zero when two complete C strings are equal.
  if (strcmp(command, "ARM") == 0) {
    // ARM permits future MOVE commands and refreshes the watchdog.
    hidArmed = true;
    lastCommandTime = millis();
    Serial.println("OK ARMED");
    return;
  }

  if (strcmp(command, "STOP") == 0) {
    // STOP releases buttons but keeps the board armed for the next MOVE.
    releaseMouseButtons();
    lastCommandTime = millis();
    Serial.println("OK STOP");
    return;
  }

  if (strcmp(command, "DISARM") == 0) {
    // DISARM blocks MOVE commands until another ARM command arrives.
    disarmHid();
    Serial.println("OK DISARMED");
    return;
  }

  // Check whether the first five characters are "MOVE ".
  if (strncmp(command, "MOVE ", 5) == 0) {
    // Skip "MOVE " and pass only the two number arguments to handleMove().
    handleMove(command + 5);
    return;
  }

  Serial.println("ERR unknown command");
}

void handleMove(const char *arguments) {
  // Convert the first text number into the horizontal movement dx.
  // afterDx points to the first character that was not part of the number.
  char *afterDx;
  long dx = strtol(arguments, &afterDx, 10);

  // A valid command must contain an integer followed by one space.
  if (afterDx == arguments || *afterDx != ' ') {
    Serial.println("ERR MOVE requires integer dx and dy");
    return;
  }

  // Skip the space and convert the second text number into vertical movement.
  const char *dyText = afterDx + 1;
  char *afterDy;
  long dy = strtol(dyText, &afterDy, 10);

  // The second integer must exist and must be the final value in the command.
  if (afterDy == dyText || *afterDy != '\0') {
    Serial.println("ERR MOVE requires integer dx and dy");
    return;
  }

  // Reject unexpectedly large movements, even if Python sends them by mistake.
  if (dx < -MAX_MOVE || dx > MAX_MOVE || dy < -MAX_MOVE || dy > MAX_MOVE) {
    Serial.println("ERR MOVE outside safety limit");
    return;
  }

  // Never move the real cursor unless Python has armed the board first.
  if (!hidArmed) {
    Serial.println("ERR DISARMED");
    return;
  }

  // The command is valid, so refresh the watchdog and move the HID mouse.
  // The third Mouse.move argument is wheel movement; zero means no scrolling.
  lastCommandTime = millis();
  Mouse.move((signed char)dx, (signed char)dy, 0);
  Serial.println("OK MOVE");
}

void releaseMouseButtons() {
  // Release every supported button to prevent a button from becoming stuck.
  Mouse.release(MOUSE_LEFT);
  Mouse.release(MOUSE_RIGHT);
  Mouse.release(MOUSE_MIDDLE);
}

void disarmHid() {
  // Put the physical output into a safe state and reject future MOVE commands.
  releaseMouseButtons();
  hidArmed = false;
}
