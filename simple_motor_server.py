#!/usr/bin/env python3
"""
Simple WebSocket Motor Server for Testing
Compatible with motor_interface.py
"""
import asyncio
import json
import serial
import websockets
from websockets.server import serve

# Configuration
PORT = 8765
AUTH_TOKEN = "robot_secret_2024"
ARDUINO_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

# Arduino connection
arduino = None

def init_arduino():
    """Initialize Arduino connection."""
    global arduino
    try:
        arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        import time
        time.sleep(2.5)  # Wait for Arduino reset
        arduino.reset_input_buffer()
        arduino.reset_output_buffer()
        print(f"✅ Arduino connected: {ARDUINO_PORT} @ {BAUD_RATE}")
        return True
    except Exception as e:
        print(f"❌ Arduino error: {e}")
        return False

def send_arduino_command(direction: str):
    """Send command to Arduino."""
    if not arduino:
        return False
    
    cmd_map = {
        "forward": "FORWARD",
        "backward": "BACKWARD",
        "left": "LEFT",
        "right": "RIGHT",
        "stop": "STOP"
    }
    
    cmd = cmd_map.get(direction.lower(), "STOP")
    try:
        arduino.write(f"{cmd}\n".encode())
        print(f"→ Arduino: {cmd}")
        return True
    except Exception as e:
        print(f"❌ Arduino send error: {e}")
        return False

async def handle_client(websocket):
    """Handle WebSocket client connection."""
    authenticated = False
    client_addr = websocket.remote_address
    
    print(f"[{client_addr}] Client connected")
    
    # Send welcome
    await websocket.send(json.dumps({
        "type": "welcome",
        "message": "Robot Motor Server v1.0"
    }))
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                cmd = data.get("cmd")
                
                # Authentication
                if cmd == "auth":
                    token = data.get("token")
                    if token == AUTH_TOKEN:
                        authenticated = True
                        await websocket.send(json.dumps({
                            "success": True,
                            "message": "Authenticated"
                        }))
                        print(f"[{client_addr}] ✅ Authenticated")
                    else:
                        await websocket.send(json.dumps({
                            "success": False,
                            "message": "Invalid token"
                        }))
                    continue
                
                if not authenticated:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Not authenticated"
                    }))
                    continue
                
                # Handle commands
                if cmd == "move":
                    direction = data.get("dir", "stop")
                    success = send_arduino_command(direction)
                    await websocket.send(json.dumps({
                        "success": success,
                        "direction": direction
                    }))
                
                elif cmd == "ping":
                    await websocket.send(json.dumps({
                        "type": "pong"
                    }))
                
                elif cmd == "status":
                    await websocket.send(json.dumps({
                        "type": "status",
                        "arduino_connected": arduino is not None,
                        "port": ARDUINO_PORT
                    }))
                
                else:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"Unknown command: {cmd}"
                    }))
            
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON"
                }))
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except websockets.ConnectionClosed:
        print(f"[{client_addr}] Client disconnected")

async def main():
    """Start server."""
    print("="*50)
    print("Simple Motor Server - Robot Pet")
    print("="*50)
    
    # Initialize Arduino
    if not init_arduino():
        print("⚠️  Continuing without Arduino (simulation mode)")
    
    print(f"\n🚀 Starting WebSocket server on port {PORT}...")
    
    async with serve(handle_client, "0.0.0.0", PORT):
        print(f"✅ Server running on ws://localhost:{PORT}")
        print("Press Ctrl+C to stop\n")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped")
        if arduino:
            arduino.close()
