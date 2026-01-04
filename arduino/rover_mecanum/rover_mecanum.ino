/*
 * Mecanum Wheel Rover - Arduino Mega
 * Holonomic drive with independent wheel control
 */

#define LF_EN   10
#define LF_IN1  26
#define LF_IN2  28
#define RF_EN   9
#define RF_IN1  22
#define RF_IN2  24
#define LR_EN   12
#define LR_IN1  7
#define LR_IN2  8
#define RR_EN   11
#define RR_IN1  5
#define RR_IN2  6

#define ENC_RIGHT_C1  2
#define ENC_RIGHT_C2  3
#define ENC_LEFT_C1   18
#define ENC_LEFT_C2   19

#define US_FL_TRIG   48
#define US_FL_ECHO   49
#define US_FR_TRIG   52
#define US_FR_ECHO   53
#define US_BACK_TRIG 50
#define US_BACK_ECHO 51

#define SPEED_DEFAULT 180

volatile long rightEncoderCount = 0;
volatile long leftEncoderCount = 0;
int wheelPWM[4] = {0, 0, 0, 0};
char cmdBuffer[64];
int cmdIndex = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }
  pinMode(LF_EN, OUTPUT); pinMode(LF_IN1, OUTPUT); pinMode(LF_IN2, OUTPUT);
  pinMode(RF_EN, OUTPUT); pinMode(RF_IN1, OUTPUT); pinMode(RF_IN2, OUTPUT);
  pinMode(LR_EN, OUTPUT); pinMode(LR_IN1, OUTPUT); pinMode(LR_IN2, OUTPUT);
  pinMode(RR_EN, OUTPUT); pinMode(RR_IN1, OUTPUT); pinMode(RR_IN2, OUTPUT);
  pinMode(ENC_RIGHT_C1, INPUT_PULLUP); pinMode(ENC_RIGHT_C2, INPUT_PULLUP);
  pinMode(ENC_LEFT_C1, INPUT_PULLUP); pinMode(ENC_LEFT_C2, INPUT_PULLUP);
  pinMode(US_FL_TRIG, OUTPUT); pinMode(US_FL_ECHO, INPUT);
  pinMode(US_FR_TRIG, OUTPUT); pinMode(US_FR_ECHO, INPUT);
  pinMode(US_BACK_TRIG, OUTPUT); pinMode(US_BACK_ECHO, INPUT);
  attachInterrupt(digitalPinToInterrupt(ENC_RIGHT_C1), rightISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_LEFT_C1), leftISR, CHANGE);
  stopMotors();
  Serial.println("ROVER:MECANUM:READY");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 10 || c == 13) {
      if (cmdIndex > 0) {
        cmdBuffer[cmdIndex] = 0;
        processCommand(cmdBuffer);
        cmdIndex = 0;
      }
    } else if (cmdIndex < 63) {
      cmdBuffer[cmdIndex++] = c;
    }
  }
}

void processCommand(char* cmd) {
  for (int i = 0; cmd[i]; i++) {
    if (cmd[i] >= 97 && cmd[i] <= 122) cmd[i] -= 32;
  }

  if (strncmp(cmd, "TWIST,", 6) == 0) {
    float vx = 0, vy = 0, wz = 0;
    if (sscanf(cmd + 6, "%f,%f,%f", &vx, &vy, &wz) == 3) {
      mecanumDrive(vx, vy, wz);
      Serial.println("OK");
    } else { Serial.println("ERROR:PARSE"); }
  }
  else if (strncmp(cmd, "MECANUM,", 8) == 0) {
    int lf, rf, lr, rr;
    if (sscanf(cmd + 8, "%d,%d,%d,%d", &lf, &rf, &lr, &rr) == 4) {
      setWheels(lf, rf, lr, rr);
      Serial.println("OK");
    } else { Serial.println("ERROR:PARSE"); }
  }
  else if (strncmp(cmd, "MOTOR,", 6) == 0) {
    int left = 0, right = 0;
    if (sscanf(cmd + 6, "%d,%d", &left, &right) == 2) {
      setWheels(left, right, left, right);
      Serial.println("OK");
    } else { Serial.println("ERROR:PARSE"); }
  }
  else if (strcmp(cmd, "FORWARD") == 0) { mecanumDrive(1.0, 0.0, 0.0); Serial.println("OK"); }
  else if (strcmp(cmd, "BACKWARD") == 0) { mecanumDrive(-1.0, 0.0, 0.0); Serial.println("OK"); }
  else if (strcmp(cmd, "LEFT") == 0) { mecanumDrive(0.0, 0.0, 1.0); Serial.println("OK"); }
  else if (strcmp(cmd, "RIGHT") == 0) { mecanumDrive(0.0, 0.0, -1.0); Serial.println("OK"); }
  else if (strcmp(cmd, "STRAFE_L") == 0) { mecanumDrive(0.0, 1.0, 0.0); Serial.println("OK"); }
  else if (strcmp(cmd, "STRAFE_R") == 0) { mecanumDrive(0.0, -1.0, 0.0); Serial.println("OK"); }
  else if (strcmp(cmd, "STOP") == 0) { stopMotors(); Serial.println("OK"); }
  else if (strcmp(cmd, "ENCODER") == 0) {
    noInterrupts(); long l = leftEncoderCount; long r = rightEncoderCount; interrupts();
    Serial.print("ENC:"); Serial.print(l); Serial.print(","); Serial.println(r);
  }
  else if (strcmp(cmd, "RESET") == 0 || strcmp(cmd, "RESET_ENCODERS") == 0) {
    noInterrupts(); leftEncoderCount = 0; rightEncoderCount = 0; interrupts();
    Serial.println("OK");
  }
  else if (strcmp(cmd, "ULTRASONIC") == 0) {
    long fl = readUS(US_FL_TRIG, US_FL_ECHO); delay(5);
    long fr = readUS(US_FR_TRIG, US_FR_ECHO); delay(5);
    long back = readUS(US_BACK_TRIG, US_BACK_ECHO);
    Serial.print("US:"); Serial.print(fl); Serial.print(","); Serial.print(fr); Serial.print(","); Serial.println(back);
  }
  else if (strcmp(cmd, "PING") == 0) { Serial.println("PONG"); }
  else if (strcmp(cmd, "STATUS") == 0) {
    Serial.print("STATUS:LF="); Serial.print(wheelPWM[0]);
    Serial.print(",RF="); Serial.print(wheelPWM[1]);
    Serial.print(",LR="); Serial.print(wheelPWM[2]);
    Serial.print(",RR="); Serial.println(wheelPWM[3]);
  }
  else { Serial.println("ERROR:UNKNOWN"); }
}

void mecanumDrive(float vx, float vy, float wz) {
  float lf = vx + vy + wz;
  float rf = vx - vy - wz;
  float lr = vx - vy + wz;
  float rr = vx + vy - wz;
  float maxVal = max(max(abs(lf), abs(rf)), max(abs(lr), abs(rr)));
  if (maxVal > 1.0) { lf /= maxVal; rf /= maxVal; lr /= maxVal; rr /= maxVal; }
  setWheels((int)(lf * SPEED_DEFAULT), (int)(rf * SPEED_DEFAULT), (int)(lr * SPEED_DEFAULT), (int)(rr * SPEED_DEFAULT));
}

void setWheels(int lf, int rf, int lr, int rr) {
  wheelPWM[0] = constrain(lf, -255, 255);
  wheelPWM[1] = constrain(rf, -255, 255);
  wheelPWM[2] = constrain(lr, -255, 255);
  wheelPWM[3] = constrain(rr, -255, 255);
  setMotor(LF_EN, LF_IN1, LF_IN2, wheelPWM[0]);
  setMotor(RF_EN, RF_IN1, RF_IN2, wheelPWM[1]);
  setMotor(LR_EN, LR_IN1, LR_IN2, wheelPWM[2]);
  setMotor(RR_EN, RR_IN1, RR_IN2, wheelPWM[3]);
}

void setMotor(int en, int in1, int in2, int pwm) {
  if (pwm > 0) { digitalWrite(in1, HIGH); digitalWrite(in2, LOW); analogWrite(en, pwm); }
  else if (pwm < 0) { digitalWrite(in1, LOW); digitalWrite(in2, HIGH); analogWrite(en, -pwm); }
  else { digitalWrite(in1, LOW); digitalWrite(in2, LOW); analogWrite(en, 0); }
}

void stopMotors() { setWheels(0, 0, 0, 0); }

void rightISR() { rightEncoderCount += (digitalRead(ENC_RIGHT_C2) == HIGH) ? 1 : -1; }
void leftISR() { leftEncoderCount += (digitalRead(ENC_LEFT_C2) == HIGH) ? 1 : -1; }

long readUS(int trig, int echo) {
  digitalWrite(trig, LOW); delayMicroseconds(2);
  digitalWrite(trig, HIGH); delayMicroseconds(10);
  digitalWrite(trig, LOW);
  long dur = pulseIn(echo, HIGH, 30000);
  return (dur == 0) ? -1 : dur / 58;
}
