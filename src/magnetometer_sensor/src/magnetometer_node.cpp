#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>

#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/magnetic_field.hpp"

class MagnetometerNode : public rclcpp::Node
{
public:
  MagnetometerNode()
  : Node("magnetometer_node"),
    i2c_fd_(-1),
    heading_logged_(false)
  {
    i2c_bus_ = this->declare_parameter<int>("i2c_bus", 1);
    i2c_address_ = this->declare_parameter<int>("i2c_address", 0x1E);
    frame_id_ = this->declare_parameter<std::string>("frame_id", "mag_link");
    publish_rate_hz_ = this->declare_parameter<double>("publish_rate_hz", 50.0);
    declination_deg_ = this->declare_parameter<double>("declination_deg", 13.5);
    x_offset_ = this->declare_parameter<double>("x_offset", 0.0);
    y_offset_ = this->declare_parameter<double>("y_offset", 0.0);
    z_offset_ = this->declare_parameter<double>("z_offset", 0.0);

    mag_pub_ = this->create_publisher<sensor_msgs::msg::MagneticField>("/imu/mag", 10);

    if (!open_i2c_device() || !configure_sensor()) {
      RCLCPP_WARN(
        this->get_logger(),
        "HMC5883L initialization failed. Node will keep running and retry reads.");
    }

    const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, publish_rate_hz_));
    timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::milliseconds>(period),
      std::bind(&MagnetometerNode::timer_callback, this));
  }

  ~MagnetometerNode() override
  {
    if (i2c_fd_ >= 0) {
      close(i2c_fd_);
      i2c_fd_ = -1;
    }
  }

private:
  bool open_i2c_device()
  {
    if (i2c_fd_ >= 0) {
      close(i2c_fd_);
      i2c_fd_ = -1;
    }

    const std::string device = "/dev/i2c-" + std::to_string(i2c_bus_);
    i2c_fd_ = open(device.c_str(), O_RDWR);
    if (i2c_fd_ < 0) {
      RCLCPP_WARN(this->get_logger(), "Failed to open %s: %s", device.c_str(), std::strerror(errno));
      return false;
    }

    if (ioctl(i2c_fd_, I2C_SLAVE, i2c_address_) < 0) {
      RCLCPP_WARN(
        this->get_logger(),
        "Failed to select I2C addr 0x%02X on %s: %s",
        i2c_address_, device.c_str(), std::strerror(errno));
      close(i2c_fd_);
      i2c_fd_ = -1;
      return false;
    }

    return true;
  }

  bool configure_sensor()
  {
    // CRA: 8 samples averaged, 75 Hz output rate, normal measurement.
    if (!write_register(0x00, 0x78)) {
      return false;
    }
    // CRB: Gain setting ±1.3 Ga.
    if (!write_register(0x01, 0x20)) {
      return false;
    }
    // Mode: Continuous-measurement mode.
    if (!write_register(0x02, 0x00)) {
      return false;
    }
    return true;
  }

  bool write_register(uint8_t reg, uint8_t value)
  {
    if (i2c_fd_ < 0) {
      return false;
    }

    uint8_t buffer[2] = {reg, value};
    const ssize_t n = write(i2c_fd_, buffer, sizeof(buffer));
    if (n != static_cast<ssize_t>(sizeof(buffer))) {
      RCLCPP_WARN(
        this->get_logger(),
        "I2C write failed reg=0x%02X value=0x%02X: %s",
        reg, value, std::strerror(errno));
      return false;
    }
    return true;
  }

  bool read_raw_xyz(int16_t & x_raw, int16_t & y_raw, int16_t & z_raw)
  {
    if (i2c_fd_ < 0 && !open_i2c_device()) {
      return false;
    }

    const uint8_t start_reg = 0x03;
    if (write(i2c_fd_, &start_reg, 1) != 1) {
      RCLCPP_WARN(this->get_logger(), "Failed to set HMC5883L read pointer: %s", std::strerror(errno));
      return false;
    }

    uint8_t data[6] = {};
    if (read(i2c_fd_, data, sizeof(data)) != static_cast<ssize_t>(sizeof(data))) {
      RCLCPP_WARN(this->get_logger(), "Failed to read HMC5883L XYZ data: %s", std::strerror(errno));
      return false;
    }

    const int16_t x = static_cast<int16_t>((data[0] << 8) | data[1]);
    const int16_t z = static_cast<int16_t>((data[2] << 8) | data[3]);
    const int16_t y = static_cast<int16_t>((data[4] << 8) | data[5]);

    x_raw = x;
    y_raw = y;
    z_raw = z;
    return true;
  }

  void timer_callback()
  {
    int16_t x_raw = 0;
    int16_t y_raw = 0;
    int16_t z_raw = 0;
    if (!read_raw_xyz(x_raw, y_raw, z_raw)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Magnetometer read failed. Check wiring/power/address (expected 0x%02X).",
        i2c_address_);
      return;
    }

    constexpr double lsb_per_gauss = 1090.0;  // For ±1.3 Ga gain setting.
    constexpr double gauss_to_tesla = 1e-4;

    const double x_cal = static_cast<double>(x_raw) - x_offset_;
    const double y_cal = static_cast<double>(y_raw) - y_offset_;
    const double z_cal = static_cast<double>(z_raw) - z_offset_;

    const double x_t = (x_cal / lsb_per_gauss) * gauss_to_tesla;
    const double y_t = (y_cal / lsb_per_gauss) * gauss_to_tesla;
    const double z_t = (z_cal / lsb_per_gauss) * gauss_to_tesla;

    sensor_msgs::msg::MagneticField msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = frame_id_;
    msg.magnetic_field.x = x_t;
    msg.magnetic_field.y = y_t;
    msg.magnetic_field.z = z_t;
    msg.magnetic_field_covariance.fill(0.0);
    mag_pub_->publish(msg);

    if (!heading_logged_) {
      constexpr double pi = 3.14159265358979323846;
      const double heading_rad = std::atan2(y_cal, x_cal) + (declination_deg_ * pi / 180.0);
      double heading_deg = heading_rad * 180.0 / pi;
      if (heading_deg < 0.0) {
        heading_deg += 360.0;
      } else if (heading_deg >= 360.0) {
        heading_deg -= 360.0;
      }
      RCLCPP_INFO(
        this->get_logger(),
        "HMC5883L online. Raw XYZ=[%d, %d, %d], heading=%.2f deg (declination %.2f deg)",
        x_raw, y_raw, z_raw, heading_deg, declination_deg_);
      heading_logged_ = true;
    }
  }

  int i2c_bus_;
  int i2c_address_;
  int i2c_fd_;
  std::string frame_id_;
  double publish_rate_hz_;
  double declination_deg_;
  double x_offset_;
  double y_offset_;
  double z_offset_;
  bool heading_logged_;

  rclcpp::Publisher<sensor_msgs::msg::MagneticField>::SharedPtr mag_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MagnetometerNode>());
  rclcpp::shutdown();
  return 0;
}
