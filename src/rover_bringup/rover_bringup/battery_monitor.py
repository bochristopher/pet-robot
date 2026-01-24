#!/usr/bin/env python3
"""
Waveshare UPS Battery Monitor for Jetson Orin
Uses INA219 power monitor on I2C Bus 7, Address 0x41

Publishes:
  - /battery_state (sensor_msgs/BatteryState)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from smbus2 import SMBus

# INA219 Register addresses
INA219_REG_CONFIG = 0x00
INA219_REG_SHUNTVOLTAGE = 0x01
INA219_REG_BUSVOLTAGE = 0x02
INA219_REG_POWER = 0x03
INA219_REG_CURRENT = 0x04
INA219_REG_CALIBRATION = 0x05

# INA219 Configuration
INA219_CONFIG_RESET = 0x8000
INA219_CONFIG_BVOLTAGERANGE_32V = 0x2000  # 0-32V range
INA219_CONFIG_GAIN_8_320MV = 0x1800  # Gain 8, 320mV range
INA219_CONFIG_BADCRES_12BIT = 0x0400  # 12-bit bus ADC resolution
INA219_CONFIG_SADCRES_12BIT = 0x0018  # 12-bit shunt ADC resolution
INA219_CONFIG_MODE_SANDBVOLT_CONTINUOUS = 0x0007

# Waveshare UPS specific
# 21700 Li-ion battery: 3.0V (empty) to 4.2V (full) per cell
# Detected 3S configuration: 9.0V (empty) to 12.6V (full)
BATTERY_VOLTAGE_MIN = 9.0   # 3S empty (3.0V/cell)
BATTERY_VOLTAGE_MAX = 12.6  # 3S full (4.2V/cell)
BATTERY_VOLTAGE_LOW = 9.6   # 3S low warning (3.2V/cell)


class BatteryMonitor(Node):
    def __init__(self):
        super().__init__('battery_monitor')
        
        # Parameters
        self.declare_parameter('i2c_bus', 7)
        self.declare_parameter('i2c_address', 0x41)
        self.declare_parameter('shunt_resistor_ohms', 0.1)  # Waveshare uses 0.1 ohm
        self.declare_parameter('max_expected_current', 3.2)  # Amps
        self.declare_parameter('publish_rate', 1.0)  # Hz
        self.declare_parameter('voltage_min', 9.0)
        self.declare_parameter('voltage_max', 12.6)
        
        self.i2c_bus = self.get_parameter('i2c_bus').value
        self.i2c_address = self.get_parameter('i2c_address').value
        self.shunt_ohms = self.get_parameter('shunt_resistor_ohms').value
        self.max_current = self.get_parameter('max_expected_current').value
        self.voltage_min = self.get_parameter('voltage_min').value
        self.voltage_max = self.get_parameter('voltage_max').value
        
        # Publisher
        self.pub = self.create_publisher(BatteryState, 'battery_state', 10)
        
        # Initialize I2C
        try:
            self.bus = SMBus(self.i2c_bus)
            self._configure_ina219()
            self.get_logger().info(
                f'Battery monitor started on I2C bus {self.i2c_bus}, address 0x{self.i2c_address:02x}'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to initialize INA219: {e}')
            self.bus = None
        
        # Timer
        publish_rate = self.get_parameter('publish_rate').value
        self.timer = self.create_timer(1.0 / publish_rate, self.timer_callback)
        
        # Track for logging
        self.last_log_time = self.get_clock().now()
        
    def _configure_ina219(self):
        """Configure INA219 for continuous voltage/current measurement"""
        config = (
            INA219_CONFIG_BVOLTAGERANGE_32V |
            INA219_CONFIG_GAIN_8_320MV |
            INA219_CONFIG_BADCRES_12BIT |
            INA219_CONFIG_SADCRES_12BIT |
            INA219_CONFIG_MODE_SANDBVOLT_CONTINUOUS
        )
        self._write_register(INA219_REG_CONFIG, config)
        
        # Calibration for current measurement
        # Cal = trunc(0.04096 / (current_lsb * shunt_ohms))
        # current_lsb = max_expected_current / 32768
        current_lsb = self.max_current / 32768.0
        cal_value = int(0.04096 / (current_lsb * self.shunt_ohms))
        self._write_register(INA219_REG_CALIBRATION, cal_value)
        
        self.current_lsb = current_lsb
        
    def _write_register(self, reg, value):
        """Write 16-bit value to register (big-endian)"""
        high = (value >> 8) & 0xFF
        low = value & 0xFF
        self.bus.write_i2c_block_data(self.i2c_address, reg, [high, low])
        
    def _read_register(self, reg):
        """Read 16-bit value from register (big-endian)"""
        data = self.bus.read_i2c_block_data(self.i2c_address, reg, 2)
        return (data[0] << 8) | data[1]
    
    def get_bus_voltage(self):
        """Get bus voltage in Volts"""
        raw = self._read_register(INA219_REG_BUSVOLTAGE)
        # Shift to remove status bits, multiply by LSB (4mV)
        voltage = ((raw >> 3) * 4) / 1000.0
        return voltage
    
    def get_shunt_voltage(self):
        """Get shunt voltage in mV"""
        raw = self._read_register(INA219_REG_SHUNTVOLTAGE)
        # Convert from two's complement if negative
        if raw > 32767:
            raw -= 65536
        # LSB is 10uV
        return raw * 0.01
    
    def get_current(self):
        """Get current in Amps"""
        raw = self._read_register(INA219_REG_CURRENT)
        # Convert from two's complement
        if raw > 32767:
            raw -= 65536
        return raw * self.current_lsb
    
    def get_power(self):
        """Get power in Watts"""
        raw = self._read_register(INA219_REG_POWER)
        # Power LSB is 20 * current_lsb
        return raw * self.current_lsb * 20
    
    def voltage_to_percentage(self, voltage):
        """Convert voltage to battery percentage"""
        if voltage >= self.voltage_max:
            return 100.0
        elif voltage <= self.voltage_min:
            return 0.0
        else:
            return ((voltage - self.voltage_min) / 
                    (self.voltage_max - self.voltage_min)) * 100.0
    
    def timer_callback(self):
        if self.bus is None:
            return
            
        try:
            voltage = self.get_bus_voltage()
            current = self.get_current()
            power = self.get_power()
            percentage = self.voltage_to_percentage(voltage)
            
            # Create BatteryState message
            msg = BatteryState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.voltage = voltage
            msg.current = current  # Positive = discharging
            msg.charge = float('nan')  # Unknown
            msg.capacity = float('nan')  # Unknown
            msg.design_capacity = float('nan')
            msg.percentage = percentage / 100.0  # 0.0 to 1.0
            msg.power_supply_status = (
                BatteryState.POWER_SUPPLY_STATUS_DISCHARGING 
                if current > 0.1 else 
                BatteryState.POWER_SUPPLY_STATUS_CHARGING
            )
            msg.power_supply_health = BatteryState.POWER_SUPPLY_HEALTH_GOOD
            msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LION
            msg.present = True
            
            self.pub.publish(msg)
            
            # Log periodically
            now = self.get_clock().now()
            if (now - self.last_log_time).nanoseconds / 1e9 > 10.0:
                status = "charging" if current < 0 else "discharging"
                self.get_logger().info(
                    f'Battery: {voltage:.2f}V ({percentage:.0f}%), '
                    f'{abs(current):.2f}A {status}, {power:.2f}W'
                )
                self.last_log_time = now
                
        except Exception as e:
            self.get_logger().error(f'Error reading battery: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = BatteryMonitor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

