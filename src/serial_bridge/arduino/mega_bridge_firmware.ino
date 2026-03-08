// mega_bridge_firmware.ino
//
// Arduino Mega 2560 - USB serial protocol bridge with REAL motor control.
// Talks directly to the Jetson over USB Serial @ 115200.
//
// Protocol (both directions): <TYPE>:<fields>,<XOR_HEX>\n
//
// RX from Jetson: M (motor), S (servo), L (LED), E (enable), D (disable),
//                 R (reset encoders), V (firmware version), H (heartbeat)
// TX to Jetson:   E (encoders), B (battery), I (IMU), K (bumper),
//                 A (heartbeat ACK), F (firmware version)

#include <Arduino.h>

// ═══════════════════════════════════════════════════════════════
// HARDWARE PIN CONFIGURATION — L298N motor drivers
// ═══════════════════════════════════════════════════════════════

// Motor pins: EN = PWM speed, IN1/IN2 = direction
constexpr uint8_t FL_EN  = 10;  constexpr uint8_t FL_IN1 = 26;  constexpr uint8_t FL_IN2 = 28;
constexpr uint8_t FR_EN  =  9;  constexpr uint8_t FR_IN1 = 22;  constexpr uint8_t FR_IN2 = 24;
constexpr uint8_t RL_EN  = 12;  constexpr uint8_t RL_IN1 =  7;  constexpr uint8_t RL_IN2 =  8;
constexpr uint8_t RR_EN  = 11;  constexpr uint8_t RR_IN1 =  5;  constexpr uint8_t RR_IN2 =  6;

// Quadrature encoders (only 2 physical encoders: left side and right side)
constexpr uint8_t ENC_RIGHT_C1 =  2;   // interrupt pin
constexpr uint8_t ENC_RIGHT_C2 =  3;
constexpr uint8_t ENC_LEFT_C1  = 18;   // interrupt pin
constexpr uint8_t ENC_LEFT_C2  = 19;

// Battery voltage divider on analog pin
constexpr uint8_t BATTERY_PIN = A0;
constexpr float BATTERY_DIVIDER_RATIO = 5.0f;  // adjust for your voltage divider

// Bumper digital input (active LOW)
constexpr uint8_t BUMPER_PIN = 30;

// ═══════════════════════════════════════════════════════════════
// PROTOCOL / TIMING CONSTANTS
// ═══════════════════════════════════════════════════════════════

constexpr uint32_t BAUD = 115200;
constexpr size_t RX_LINE_MAX = 160;
constexpr uint32_t TELEMETRY_FAST_MS  = 50;   // encoders + IMU at 20 Hz
constexpr uint32_t TELEMETRY_SLOW_MS  = 500;  // battery + bumper at 2 Hz
constexpr uint32_t MOTOR_TIMEOUT_MS   = 1000; // zero motors if no M: command for 1s

const char *FW_VERSION = "mega-usb-v3.0";

// ═══════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════

volatile bool motors_enabled = false;
int16_t pwm_fl = 0, pwm_fr = 0, pwm_rl = 0, pwm_rr = 0;

// Encoders: left pair drives FL+RL, right pair drives FR+RR
volatile long enc_left  = 0;
volatile long enc_right = 0;

uint32_t hb_counter = 0;
uint32_t last_motor_cmd_ms = 0;

char rx_buf[RX_LINE_MAX];
size_t rx_len = 0;
uint32_t last_fast_ms = 0;
uint32_t last_slow_ms = 0;

// ═══════════════════════════════════════════════════════════════
// MOTOR DRIVER — L298N: EN=PWM, IN1/IN2=direction
// ═══════════════════════════════════════════════════════════════

void setMotor(uint8_t en, uint8_t in1, uint8_t in2, int16_t pwm) {
  int16_t duty = constrain(abs(pwm), 0, 255);
  if (pwm > 0) {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    analogWrite(en, duty);
  } else if (pwm < 0) {
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    analogWrite(en, duty);
  } else {
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    analogWrite(en, 0);
  }
}

void applyMotors() {
  if (!motors_enabled) {
    setMotor(FL_EN, FL_IN1, FL_IN2, 0);
    setMotor(FR_EN, FR_IN1, FR_IN2, 0);
    setMotor(RL_EN, RL_IN1, RL_IN2, 0);
    setMotor(RR_EN, RR_IN1, RR_IN2, 0);
    return;
  }
  setMotor(FL_EN, FL_IN1, FL_IN2, pwm_fl);
  setMotor(FR_EN, FR_IN1, FR_IN2, pwm_fr);
  setMotor(RL_EN, RL_IN1, RL_IN2, pwm_rl);
  setMotor(RR_EN, RR_IN1, RR_IN2, pwm_rr);
}

void stopMotors() {
  pwm_fl = pwm_fr = pwm_rl = pwm_rr = 0;
  applyMotors();
}

// ═══════════════════════════════════════════════════════════════
// ENCODER ISRs — quadrature with direction detection
// ═══════════════════════════════════════════════════════════════

void rightISR() {
  enc_right += (digitalRead(ENC_RIGHT_C2) == HIGH) ? 1 : -1;
}

void leftISR() {
  enc_left += (digitalRead(ENC_LEFT_C2) == HIGH) ? 1 : -1;
}

// ═══════════════════════════════════════════════════════════════
// CHECKSUM + FRAMING
// ═══════════════════════════════════════════════════════════════

uint8_t xorChecksum(const char *data, size_t len) {
  uint8_t c = 0;
  for (size_t i = 0; i < len; ++i) c ^= static_cast<uint8_t>(data[i]);
  return c;
}

void toHex2(uint8_t b, char out[3]) {
  static const char *HEX_DIGITS = "0123456789ABCDEF";
  out[0] = HEX_DIGITS[(b >> 4) & 0x0F];
  out[1] = HEX_DIGITS[b & 0x0F];
  out[2] = '\0';
}

bool parseHex2(const char *s, uint8_t &out) {
  if (!s || strlen(s) != 2) return false;
  auto nib = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  int hi = nib(s[0]);
  int lo = nib(s[1]);
  if (hi < 0 || lo < 0) return false;
  out = static_cast<uint8_t>((hi << 4) | lo);
  return true;
}

void sendPayload(const char *payload) {
  char cs[3];
  toHex2(xorChecksum(payload, strlen(payload)), cs);
  Serial.print(payload);
  Serial.print(",");
  Serial.print(cs);
  Serial.print('\n');
}

// ═══════════════════════════════════════════════════════════════
// TELEMETRY SENDERS
// ═══════════════════════════════════════════════════════════════

void sendEncoders() {
  noInterrupts();
  long l = enc_left;
  long r = enc_right;
  interrupts();
  // Report as 4 wheels: FL=left, FR=right, RL=left, RR=right
  char p[96];
  snprintf(p, sizeof(p), "E:%ld,%ld,%ld,%ld", l, r, l, r);
  sendPayload(p);
}

void sendBattery() {
  int raw = analogRead(BATTERY_PIN);
  float voltage = (raw / 1023.0f) * 5.0f * BATTERY_DIVIDER_RATIO;
  char vstr[16];
  dtostrf(voltage, 1, 2, vstr);
  char payload[48];
  snprintf(payload, sizeof(payload), "B:%s", vstr);
  sendPayload(payload);
}

void sendImu() {
  // Placeholder — add MPU6050 I2C reads when wired
  sendPayload("I:0.00,0.00,9.81,0.00,0.00,0.00");
}

void sendBumper() {
  uint8_t state = (digitalRead(BUMPER_PIN) == LOW) ? 1 : 0;
  char p[12];
  snprintf(p, sizeof(p), "K:%u", state);
  sendPayload(p);
}

void sendHeartbeatAck() {
  char p[24];
  snprintf(p, sizeof(p), "A:%lu", (unsigned long)hb_counter);
  sendPayload(p);
}

void sendFirmware() {
  char p[80];
  snprintf(p, sizeof(p), "F:%s", FW_VERSION);
  sendPayload(p);
}

// ═══════════════════════════════════════════════════════════════
// PARSING HELPERS
// ═══════════════════════════════════════════════════════════════

bool parseInt16Safe(const char *s, int16_t &out) {
  if (!s || !*s) return false;
  char *end = nullptr;
  long v = strtol(s, &end, 10);
  if (end == s || *end != '\0') return false;
  if (v < -32768 || v > 32767) return false;
  out = static_cast<int16_t>(v);
  return true;
}

size_t splitCsv(char *s, char *tokens[], size_t max_tokens) {
  if (!s || !*s) return 0;
  size_t n = 0;
  char *p = s;
  while (n < max_tokens) {
    tokens[n++] = p;
    char *c = strchr(p, ',');
    if (!c) break;
    *c = '\0';
    p = c + 1;
  }
  return n;
}

bool verifyAndExtractPayload(char *line) {
  size_t n = strlen(line);
  if (n < 4) return false;
  const char *cs_text = line + n - 2;
  uint8_t expected = 0;
  if (!parseHex2(cs_text, expected)) return false;
  line[n - 2] = '\0';
  size_t payload_len = strlen(line);
  if (payload_len == 0) return false;
  if (line[payload_len - 1] == ',') line[payload_len - 1] = '\0';
  uint8_t actual = xorChecksum(line, strlen(line));
  return actual == expected;
}

// ═══════════════════════════════════════════════════════════════
// COMMAND HANDLER
// ═══════════════════════════════════════════════════════════════

void handlePayload(char *payload) {
  if (!payload || strlen(payload) < 2 || payload[1] != ':') return;
  char type = payload[0];
  char *fields = payload + 2;

  switch (type) {
    case 'M': {
      char *t[4] = {nullptr, nullptr, nullptr, nullptr};
      if (splitCsv(fields, t, 4) != 4) return;
      int16_t fl, fr, rl, rr;
      if (!parseInt16Safe(t[0], fl) || !parseInt16Safe(t[1], fr) ||
          !parseInt16Safe(t[2], rl) || !parseInt16Safe(t[3], rr)) return;
      if (!motors_enabled) { stopMotors(); return; }
      pwm_fl = fl; pwm_fr = fr; pwm_rl = rl; pwm_rr = rr;
      applyMotors();
      last_motor_cmd_ms = millis();
      break;
    }
    case 'S': {
      char *t[2] = {nullptr, nullptr};
      if (splitCsv(fields, t, 2) != 2) break;
      // Servo control — add Servo.write() calls when servos are wired
      break;
    }
    case 'L': break;  // LED control — add NeoPixel/LED calls when wired
    case 'E': motors_enabled = true; stopMotors(); last_motor_cmd_ms = millis(); break;
    case 'D': motors_enabled = false; stopMotors(); break;
    case 'R': noInterrupts(); enc_left = 0; enc_right = 0; interrupts(); break;
    case 'V': sendFirmware(); break;
    case 'H': hb_counter++; sendHeartbeatAck(); break;
    default: break;
  }
}

void processLine(char *line) {
  if (!verifyAndExtractPayload(line)) return;
  handlePayload(line);
}

// ═══════════════════════════════════════════════════════════════
// SERIAL RX
// ═══════════════════════════════════════════════════════════════

void pollSerial() {
  while (Serial.available() > 0) {
    char ch = static_cast<char>(Serial.read());
    if (ch == '\r') continue;
    if (ch == '\n') {
      if (rx_len > 0) {
        rx_buf[rx_len] = '\0';
        processLine(rx_buf);
        rx_len = 0;
      }
    } else if (rx_len < RX_LINE_MAX - 1) {
      rx_buf[rx_len++] = ch;
    } else {
      rx_len = 0;  // overflow — discard
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// SAFETY — local motor timeout (defense in depth)
// ═══════════════════════════════════════════════════════════════

void checkMotorTimeout() {
  if (motors_enabled && (millis() - last_motor_cmd_ms > MOTOR_TIMEOUT_MS)) {
    stopMotors();
  }
}

// ═══════════════════════════════════════════════════════════════
// SETUP + LOOP
// ═══════════════════════════════════════════════════════════════

void setup() {
  Serial.begin(BAUD);

  // Motor driver pins
  pinMode(FL_EN,  OUTPUT); pinMode(FL_IN1, OUTPUT); pinMode(FL_IN2, OUTPUT);
  pinMode(FR_EN,  OUTPUT); pinMode(FR_IN1, OUTPUT); pinMode(FR_IN2, OUTPUT);
  pinMode(RL_EN,  OUTPUT); pinMode(RL_IN1, OUTPUT); pinMode(RL_IN2, OUTPUT);
  pinMode(RR_EN,  OUTPUT); pinMode(RR_IN1, OUTPUT); pinMode(RR_IN2, OUTPUT);

  // Encoder pins
  pinMode(ENC_RIGHT_C1, INPUT_PULLUP); pinMode(ENC_RIGHT_C2, INPUT_PULLUP);
  pinMode(ENC_LEFT_C1,  INPUT_PULLUP); pinMode(ENC_LEFT_C2,  INPUT_PULLUP);

  // Bumper
  pinMode(BUMPER_PIN, INPUT_PULLUP);

  // Attach encoder interrupts
  attachInterrupt(digitalPinToInterrupt(ENC_RIGHT_C1), rightISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_LEFT_C1),  leftISR,  CHANGE);

  stopMotors();
  last_motor_cmd_ms = millis();

  delay(200);
}

void loop() {
  pollSerial();
  checkMotorTimeout();

  uint32_t now = millis();
  if ((now - last_fast_ms) >= TELEMETRY_FAST_MS) {
    last_fast_ms = now;
    sendEncoders();
    sendImu();
  }
  if ((now - last_slow_ms) >= TELEMETRY_SLOW_MS) {
    last_slow_ms = now;
    sendBattery();
    sendBumper();
  }
}
