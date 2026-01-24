#!/bin/bash
# Quick battery status check using I2C directly
# Waveshare UPS with INA219 on I2C Bus 7, Address 0x41

I2C_BUS=7
I2C_ADDR=0x41

# Read bus voltage register (0x02)
raw=$(i2cget -y $I2C_BUS $I2C_ADDR 0x02 w 2>/dev/null)

if [ -z "$raw" ]; then
    echo "❌ Battery monitor not detected!"
    exit 1
fi

# Convert: swap bytes, shift right 3, multiply by 4mV
# The i2cget returns little-endian, need to swap
hex_val=${raw#0x}
byte1=${hex_val:0:2}
byte2=${hex_val:2:2}
swapped=$((16#$byte2$byte1))
voltage_raw=$(( (swapped >> 3) * 4 ))
voltage=$(echo "scale=2; $voltage_raw / 1000" | bc)

# Calculate percentage (3S: 9.0V=0%, 12.6V=100%)
V_MIN=9.0
V_MAX=12.6
percentage=$(python3 -c "v=$voltage; vmin=$V_MIN; vmax=$V_MAX; p=max(0,min(100,int((v-vmin)/(vmax-vmin)*100))); print(p)" 2>/dev/null)

# Default if calculation fails
if [ -z "$percentage" ]; then
    percentage="??"
fi

# Display with color
if [ "$percentage" -gt 50 ]; then
    color="\033[32m"  # Green
elif [ "$percentage" -gt 20 ]; then
    color="\033[33m"  # Yellow
else
    color="\033[31m"  # Red
fi
reset="\033[0m"

echo -e "🔋 Battery: ${color}${voltage}V (${percentage}%)${reset}"

# Show charging status via current
raw_current=$(i2cget -y $I2C_BUS $I2C_ADDR 0x04 w 2>/dev/null)
if [ -n "$raw_current" ]; then
    hex_curr=${raw_current#0x}
    b1=${hex_curr:0:2}
    b2=${hex_curr:2:2}
    curr_raw=$((16#$b2$b1))
    # Two's complement for negative
    if [ $curr_raw -gt 32767 ]; then
        curr_raw=$((curr_raw - 65536))
    fi
    # Current LSB depends on calibration, approximate
    if [ $curr_raw -lt -100 ]; then
        echo "   ⚡ Status: Charging"
    elif [ $curr_raw -gt 100 ]; then
        echo "   🔌 Status: Discharging"
    else
        echo "   ✓ Status: Idle/Full"
    fi
fi

