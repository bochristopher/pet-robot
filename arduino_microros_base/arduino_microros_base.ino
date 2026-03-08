#include <micro_ros_arduino.h>

#include <geometry_msgs/msg/twist.h>
#include <std_msgs/msg/int32_multi_array.h>

#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <rmw_microros/rmw_microros.h>

#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if ((temp_rc != RCL_RET_OK)) { return false; } }
#define EXECUTE_EVERY_N_MS(MS, X) do { \
  static volatile int64_t init = -1;   \
  if (init == -1) { init = uxr_millis(); } \
  if (uxr_millis() - init > MS) { X; init = uxr_millis(); } \
} while (0)

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

float last_linear_x = 0.0f;
float last_angular_z = 0.0f;

enum states {
  WAITING_AGENT,
  AGENT_AVAILABLE,
  AGENT_CONNECTED,
  AGENT_DISCONNECTED
} state;

void cmd_vel_callback(const void * msg_in) {
  const auto * msg = static_cast<const geometry_msgs__msg__Twist *>(msg_in);
  last_linear_x = msg->linear.x;
  last_angular_z = msg->angular.z;
}

void encoder_timer_callback(rcl_timer_t * timer, int64_t last_call_time) {
  (void) last_call_time;
  if (timer == nullptr) {
    return;
  }

  // Temporary synthetic encoder update to validate end-to-end ROS path.
  encoder_data[0] += static_cast<int32_t>(last_linear_x * 10.0f - last_angular_z * 3.0f);
  encoder_data[1] += static_cast<int32_t>(last_linear_x * 10.0f + last_angular_z * 3.0f);

  encoder_msg.data.data = encoder_data;
  encoder_msg.data.size = 2;
  encoder_msg.data.capacity = 2;
  rcl_publish(&encoder_pub, &encoder_msg, nullptr);
}

bool create_entities() {
  allocator = rcl_get_default_allocator();

  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));
  RCCHECK(rclc_node_init_default(&node, "esp32_base_node", "", &support));

  RCCHECK(rclc_subscription_init_default(
    &cmd_vel_sub,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Twist),
    "/cmd_vel"));

  RCCHECK(rclc_publisher_init_best_effort(
    &encoder_pub,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32MultiArray),
    "/wheel_encoders"));

  RCCHECK(rclc_timer_init_default(
    &encoder_timer,
    &support,
    RCL_MS_TO_NS(50),
    encoder_timer_callback));

  executor = rclc_executor_get_zero_initialized_executor();
  RCCHECK(rclc_executor_init(&executor, &support.context, 2, &allocator));
  RCCHECK(rclc_executor_add_subscription(
    &executor,
    &cmd_vel_sub,
    &cmd_vel_msg,
    &cmd_vel_callback,
    ON_NEW_DATA));
  RCCHECK(rclc_executor_add_timer(&executor, &encoder_timer));

  return true;
}

void destroy_entities() {
  rmw_context_t * rmw_context = rcl_context_get_rmw_context(&support.context);
  (void) rmw_uros_set_context_entity_destroy_session_timeout(rmw_context, 0);

  rcl_subscription_fini(&cmd_vel_sub, &node);
  rcl_publisher_fini(&encoder_pub, &node);
  rcl_timer_fini(&encoder_timer);
  rclc_executor_fini(&executor);
  rcl_node_fini(&node);
  rclc_support_fini(&support);
}

void setup() {
  set_microros_transports();
  delay(2000);
  state = WAITING_AGENT;
}

void loop() {
  switch (state) {
    case WAITING_AGENT:
      EXECUTE_EVERY_N_MS(500, state = (RMW_RET_OK == rmw_uros_ping_agent(100, 1)) ? AGENT_AVAILABLE : WAITING_AGENT;);
      break;
    case AGENT_AVAILABLE:
      state = (true == create_entities()) ? AGENT_CONNECTED : WAITING_AGENT;
      if (state == WAITING_AGENT) {
        destroy_entities();
      }
      break;
    case AGENT_CONNECTED:
      EXECUTE_EVERY_N_MS(250, state = (RMW_RET_OK == rmw_uros_ping_agent(100, 1)) ? AGENT_CONNECTED : AGENT_DISCONNECTED;);
      if (state == AGENT_CONNECTED) {
        rclc_executor_spin_some(&executor, RCL_MS_TO_NS(50));
      }
      break;
    case AGENT_DISCONNECTED:
      destroy_entities();
      state = WAITING_AGENT;
      break;
    default:
      break;
  }

  delay(10);
}
