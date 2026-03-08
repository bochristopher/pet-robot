#include <micro_ros_arduino.h>
#include <geometry_msgs/msg/twist.h>
#include <std_msgs/msg/int32_multi_array.h>

#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>

// ---------------- Pin configuration (edit for your wiring) ----------------
constexpr int LEFT_PWM_PIN = 5;
constexpr int LEFT_DIR_PIN = 22;
constexpr int RIGHT_PWM_PIN = 6;
constexpr int RIGHT_DIR_PIN = 23;

constexpr int LEFT_ENCODER_PIN = 2;   // external interrupt pin
constexpr int RIGHT_ENCODER_PIN = 3;  // external interrupt pin

// ---------------- Robot tuning (edit for your robot) ----------------
constexpr float WHEEL_BASE_M = 0.2286f;
constexpr float MAX_LINEAR_MPS = 0.5f;
constexpr int MAX_PWM = 255;
constexpr unsigned long ENCODER_PUBLISH_MS = 20;

rcl_allocator_t allocator;
rclc_support_t support;
rcl_node_t node;
rcl_subscription_t cmd_vel_sub;
rcl_publisher_t encoder_pub;
rcl_timer_t encoder_timer;
rclc_executor_t executor;

geometry_msgs__msg__Twist cmd_vel_msg;
std_msgs__msg__Int32MultiArray encoder_msg;
int32_t encoder_data[2] = {0, 0};

volatile int32_t left_ticks = 0;
volatile int32_t right_ticks = 0;

#define RCCHECK(fn) { rcl_ret_t rc = fn; if ((rc != RCL_RET_OK)) { error_loop(); } }
#define RCSOFTCHECK(fn) { rcl_ret_t rc = fn; (void)rc; }

void left_encoder_isr() {
  left_ticks++;
}

void right_encoder_isr() {
  right_ticks++;
}

void error_loop() {
  while (1) {
    delay(100);
  }
}

int clamp_pwm(int pwm) {
  if (pwm > MAX_PWM) return MAX_PWM;
  if (pwm < -MAX_PWM) return -MAX_PWM;
  return pwm;
}

void apply_motor(int pwm, int dir_pin, int pwm_pin) {
  bool forward = pwm >= 0;
  int duty = abs(pwm);
  digitalWrite(dir_pin, forward ? HIGH : LOW);
  analogWrite(pwm_pin, duty);
}

void drive_from_cmd_vel(float linear_x, float angular_z) {
  // Differential conversion from Twist to wheel target speed.
  float left_mps = linear_x - (angular_z * WHEEL_BASE_M * 0.5f);
  float right_mps = linear_x + (angular_z * WHEEL_BASE_M * 0.5f);

  int left_pwm = static_cast<int>((left_mps / MAX_LINEAR_MPS) * MAX_PWM);
  int right_pwm = static_cast<int>((right_mps / MAX_LINEAR_MPS) * MAX_PWM);

  left_pwm = clamp_pwm(left_pwm);
  right_pwm = clamp_pwm(right_pwm);

  apply_motor(left_pwm, LEFT_DIR_PIN, LEFT_PWM_PIN);
  apply_motor(right_pwm, RIGHT_DIR_PIN, RIGHT_PWM_PIN);
}

void cmd_vel_callback(const void * msg_in) {
  const auto * msg = static_cast<const geometry_msgs__msg__Twist *>(msg_in);
  drive_from_cmd_vel(msg->linear.x, msg->angular.z);
}

void encoder_timer_callback(rcl_timer_t * timer, int64_t last_call_time) {
  (void) last_call_time;
  if (timer == nullptr) {
    return;
  }

  noInterrupts();
  encoder_data[0] = left_ticks;
  encoder_data[1] = right_ticks;
  interrupts();

  encoder_msg.data.data = encoder_data;
  encoder_msg.data.size = 2;
  encoder_msg.data.capacity = 2;
  RCSOFTCHECK(rcl_publish(&encoder_pub, &encoder_msg, nullptr));
}

void setup() {
  pinMode(LEFT_PWM_PIN, OUTPUT);
  pinMode(LEFT_DIR_PIN, OUTPUT);
  pinMode(RIGHT_PWM_PIN, OUTPUT);
  pinMode(RIGHT_DIR_PIN, OUTPUT);

  pinMode(LEFT_ENCODER_PIN, INPUT_PULLUP);
  pinMode(RIGHT_ENCODER_PIN, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(LEFT_ENCODER_PIN), left_encoder_isr, RISING);
  attachInterrupt(digitalPinToInterrupt(RIGHT_ENCODER_PIN), right_encoder_isr, RISING);

  set_microros_transports();
  delay(2000);

  allocator = rcl_get_default_allocator();
  RCCHECK(rclc_support_init(&support, 0, nullptr, &allocator));
  RCCHECK(rclc_node_init_default(&node, "arduino_base_node", "", &support));

  RCCHECK(rclc_subscription_init_default(
    &cmd_vel_sub,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Twist),
    "/cmd_vel"));

  RCCHECK(rclc_publisher_init_default(
    &encoder_pub,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32MultiArray),
    "/wheel_encoders"));

  RCCHECK(rclc_timer_init_default(
    &encoder_timer,
    &support,
    RCL_MS_TO_NS(ENCODER_PUBLISH_MS),
    encoder_timer_callback));

  RCCHECK(rclc_executor_init(&executor, &support.context, 2, &allocator));
  RCCHECK(rclc_executor_add_subscription(
    &executor,
    &cmd_vel_sub,
    &cmd_vel_msg,
    &cmd_vel_callback,
    ON_NEW_DATA));
  RCCHECK(rclc_executor_add_timer(&executor, &encoder_timer));
}

void loop() {
  RCSOFTCHECK(rclc_executor_spin_some(&executor, RCL_MS_TO_NS(5)));
  delay(5);
}
