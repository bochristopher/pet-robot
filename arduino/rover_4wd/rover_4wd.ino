/*
 * Robot Motor Control for Arduino Mega
 * 4-wheel drive with encoder feedback
 * ROS2 Compatible with PWM control
 *
 * Commands:
 *   MOTOR,left,right  - PWM control (-255 to 255) for ROS2 nav2
 *   FORWARD, BACKWARD, LEFT, RIGHT, STOP - Direction commands
 *   ENCODER           - Get encoder counts (ENC:left,right)
 *   ULTRASONIC        - Get distances (US:fl,fr,back)
 *   RESET             - Reset encoders
 *   PING              - Returns PONG
 */

// Left Front Motor
#define LF_EN   10
#define LF_IN1  26
#define LF_IN2  28

// Right Front Motor
#define RF_EN   9
#define RF_IN1  22
#define RF_IN2  24

// Left Rear Motor
#define LR_EN   12
#define LR_IN1  7
#define LR_IN2  8

// Right Rear Motor
#define RR_EN   11
#define RR_IN1  5
#define RR_IN2  6

// Encoder pins (back wheels only)
#define ENC_RIGHT_C1  2
#define ENC_RIGHT_C2  3
#define ENC_LEFT_C1   18
#define ENC_LEFT_C2   19

// Ultrasonic sensor pins
#define US_FL_TRIG   48
#define US_FL_ECHO   49
#define US_FR_TRIG   52
#define US_FR_ECHO   53
#define US_BACK_TRIG 50
#define US_BACK_ECHO 51

#define SPEED_DEFAULT 200
#define SPEED_TURN    180

// Encoder variables
volatile long rightEncoderCount = 0;
volatile long leftEncoderCount = 0;

// Current motor state
int currentLeftPWM = 0;
int currentRightPWM = 0;

// Command buffer
char cmdBuffer[64];
int cmdIndex = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  // Motor pins
  pinMode(LF_EN, OUTPUT); pinMode(LF_IN1, OUTPUT); pinMode(LF_IN2, OUTPUT);
  pinMode(RF_EN, OUTPUT); pinMode(RF_IN1, OUTPUT); pinMode(RF_IN2, OUTPUT);
  pinMode(LR_EN, OUTPUT); pinMode(LR_IN1, OUTPUT); pinMode(LR_IN2, OUTPUT);
  pinMode(RR_EN, OUTPUT); pinMode(RR_IN1, OUTPUT); pinMode(RR_IN2, OUTPUT);

  // Encoder pins
  pinMode(ENC_RIGHT_C1, INPUT_PULLUP);
  pinMode(ENC_RIGHT_C2, INPUT_PULLUP);
  pinMode(ENC_LEFT_C1, INPUT_PULLUP);
  pinMode(ENC_LEFT_C2, INPUT_PULLUP);

  // Ultrasonic pins
  pinMode(US_FL_TRIG, OUTPUT); pinMode(US_FL_ECHO, INPUT);
  pinMode(US_FR_TRIG, OUTPUT); pinMode(US_FR_ECHO, INPUT);
  pinMode(US_BACK_TRIG, OUTPUT); pinMode(US_BACK_ECHO, INPUT);

  // Encoder interrupts
  attachInterrupt(digitalPinToInterrupt(ENC_RIGHT_C1), rightISR_C1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_RIGHT_C2), rightISR_C2, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_LEFT_C1), leftISR_C1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_LEFT_C2), leftISR_C2, CHANGE);

  stopMotors();
  Serial.println("ROVER:READY");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (cmdIndex > 0) {
        cmdBuffer[cmdIndex] = '\0';
        processCommand(cmdBuffer);
        cmdIndex = 0;
      }
    } else if (cmdIndex < 63) {
      cmdBuffer[cmdIndex++] = c;
    }
  }
}

void processCommand(char* cmd) {
  // Convert to uppercase
  for (int i = 0; cmd[i]; i++) {
    if (cmd[i] >= 'a' && cmd[i] <= 'z') cmd[i] -= 32;
  }

  // PWM Motor command: MOTOR,left,right
  if (strncmp(cmd, "MOTOR,", 6) == 0) {
    int left = 0, right = 0;
    if (sscanf(cmd + 6, "%d,%d", &left, &right) == 2) {
      setMotorsPWM(left, right);
      Serial.println("OK");
    } else {
      Serial.println("ERROR:PARSE");
    }
  }
  // Direction commands
  else if (strcmp(cmd, "FORWARD") == 0) {
    setMotorsPWM(SPEED_DEFAULT, SPEED_DEFAULT);
    Serial.println("OK");
  }
  else if (strcmp(cmd, "BACKWARD") == 0) {
    setMotorsPWM(-SPEED_DEFAULT, -SPEED_DEFAULT);
    Serial.println("OK");
  }
  else if (strcmp(cmd, "LEFT") == 0) {
    setMotorsPWM(-SPEED_TURN, SPEED_TURN);
    Serial.println("OK");
  }
  else if (strcmp(cmd, "RIGHT") == 0) {
    setMotorsPWM(SPEED_TURN, -SPEED_TURN);
    Serial.println("OK");
  }
  else if (strcmp(cmd, "STOP") == 0) {
    stopMotors();
    Serial.println("OK");
  }
  // Sensor commands
  else if (strcmp(cmd, "ENCODER") == 0) {
    noInterrupts();
    long left = leftEncoderCount;
    long right = rightEncoderCount;
    interrupts();
    Serial.print("ENC:");
    Serial.print(left);
    Serial.print(",");
    Serial.println(right);
  }
  else if (strcmp(cmd, "RESET") == 0 || strcmp(cmd, "RESET_ENCODERS") == 0) {
    noInterrupts();
    leftEncoderCount = 0;
    rightEncoderCount = 0;
    interrupts();
    Serial.println("OK");
  }
  else if (strcmp(cmd, "ULTRASONIC") == 0) {
    long fl = readUS(US_FL_TRIG, US_FL_ECHO);
    delay(5);
    long fr = readUS(US_FR_TRIG, US_FR_ECHO);
    delay(5);
    long back = readUS(US_BACK_TRIG, US_BACK_ECHO);
    Serial.print("US:");
    Serial.print(fl);
    Serial.print(",");
    Serial.print(fr);
    Serial.print(",");
    Serial.println(back);
  }
  else if (strcmp(cmd, "PING") == 0) {
    Serial.println("PONG");
  }
  else if (strcmp(cmd, "STATUS") == 0) {
    Serial.print("STATUS:L=");
    Serial.print(currentLeftPWM);
    Serial.print(",R=");
    Serial.println(currentRightPWM);
  }
  else {
    Serial.println("ERROR:UNKNOWN");
  }
}

// Set all 4 motors with PWM (-255 to 255)
void setMotorsPWM(int left, int right) {
  left = constrain(left, -255, 255);
  right = constrain(right, -255, 255);
  currentLeftPWM = left;
  currentRightPWM = right;

  // Left side (front + rear)
  if (left > 0) {
    digitalWrite(LF_IN1, HIGH); digitalWrite(LF_IN2, LOW);
    digitalWrite(LR_IN1, HIGH); digitalWrite(LR_IN2, LOW);
    analogWrite(LF_EN, left);
    analogWrite(LR_EN, left);
  } else if (left < 0) {
    digitalWrite(LF_IN1, LOW); digitalWrite(LF_IN2, HIGH);
    digitalWrite(LR_IN1, LOW); digitalWrite(LR_IN2, HIGH);
    analogWrite(LF_EN, -left);
    analogWrite(LR_EN, -left);
  } else {
    digitalWrite(LF_IN1, LOW); digitalWrite(LF_IN2, LOW);
    digitalWrite(LR_IN1, LOW); digitalWrite(LR_IN2, LOW);
    analogWrite(LF_EN, 0);
    analogWrite(LR_EN, 0);
  }

  // Right side (front + rear)
  if (right > 0) {
    digitalWrite(RF_IN1, HIGH); digitalWrite(RF_IN2, LOW);
    digitalWrite(RR_IN1, HIGH); digitalWrite(RR_IN2, LOW);
    analogWrite(RF_EN, right);
    analogWrite(RR_EN, right);
  } else if (right < 0) {
    digitalWrite(RF_IN1, LOW); digitalWrite(RF_IN2, HIGH);
    digitalWrite(RR_IN1, LOW); digitalWrite(RR_IN2, HIGH);
    analogWrite(RF_EN, -right);
    analogWrite(RR_EN, -right);
  } else {
    digitalWrite(RF_IN1, LOW); digitalWrite(RF_IN2, LOW);
    digitalWrite(RR_IN1, LOW); digitalWrite(RR_IN2, LOW);
    analogWrite(RF_EN, 0);
    analogWrite(RR_EN, 0);
  }
}

void stopMotors() {
  setMotorsPWM(0, 0);
}

// Encoder ISRs (quadrature decoding)
void rightISR_C1() {
  rightEncoderCount += (digitalRead(ENC_RIGHT_C1) == digitalRead(ENC_RIGHT_C2)) ? 1 : -1;
}
void rightISR_C2() {
  rightEncoderCount += (digitalRead(ENC_RIGHT_C1) != digitalRead(ENC_RIGHT_C2)) ? 1 : -1;
}
void leftISR_C1() {
  leftEncoderCount += (digitalRead(ENC_LEFT_C1) == digitalRead(ENC_LEFT_C2)) ? 1 : -1;
}
void leftISR_C2() {
  leftEncoderCount += (digitalRead(ENC_LEFT_C1) != digitalRead(ENC_LEFT_C2)) ? 1 : -1;
}

// Ultrasonic reading
long readUS(int trig, int echo) {
  digitalWrite(trig, LOW);
  delayMicroseconds(2);
  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);

  long dur = pulseIn(echo, HIGH, 30000);
  if (dur == 0) return -1;
  return dur / 58;  // cm
}
