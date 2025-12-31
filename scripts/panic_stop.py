#!/usr/bin/env python3
"""
PANIC STOP - Emergency Robot Stop Web Interface

Ultra-reliable, minimal-dependency stop button that sends STOP
directly to Arduino via serial, bypassing all other layers.

Run: python3 panic_stop.py
Open: http://<jetson-ip>:5000
"""

import os
import sys
import time
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler

# Try to import serial
try:
    import serial
except ImportError:
    print("Installing pyserial...")
    os.system("pip3 install pyserial")
    import serial

# Configuration
PORT = 5000
SERIAL_PORT = "/dev/ttyACM0"  # Arduino Mega
SERIAL_BAUD = 115200

# Global serial connection
arduino = None

def get_ip():
    """Get local IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

def connect_arduino():
    """Connect to Arduino."""
    global arduino
    ports = ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0", "/dev/ttyUSB1"]

    for port in ports:
        try:
            arduino = serial.Serial(port, SERIAL_BAUD, timeout=1)
            time.sleep(0.5)  # Wait for Arduino reset
            print(f"[PANIC] Connected to Arduino on {port}")
            return True
        except:
            continue

    print("[PANIC] WARNING: No Arduino found - stop may not work!")
    return False

def send_stop():
    """Send STOP command to Arduino."""
    global arduino

    try:
        if arduino is None or not arduino.is_open:
            connect_arduino()

        if arduino and arduino.is_open:
            arduino.write(b"STOP\n")
            arduino.flush()
            # Read response
            time.sleep(0.1)
            if arduino.in_waiting:
                resp = arduino.readline().decode().strip()
                return f"OK: {resp}"
            return "OK: STOP sent"
    except Exception as e:
        # Try to reconnect
        try:
            connect_arduino()
            if arduino:
                arduino.write(b"STOP\n")
                return "OK: Reconnected and stopped"
        except:
            pass
        return f"ERROR: {e}"

    return "ERROR: No Arduino connection"

# HTML for the panic button
HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>PANIC STOP</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        html, body {
            height: 100%;
            width: 100%;
            background: #1a1a1a;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            overflow: hidden;
            touch-action: manipulation;
        }

        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            padding: 20px;
        }

        #stop-btn {
            width: min(90vw, 90vh, 500px);
            height: min(90vw, 90vh, 500px);
            border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, #ff4444, #cc0000 50%, #990000);
            border: 8px solid #660000;
            box-shadow:
                0 10px 30px rgba(0,0,0,0.5),
                inset 0 -5px 20px rgba(0,0,0,0.3),
                inset 0 5px 20px rgba(255,255,255,0.1);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.1s, box-shadow 0.1s;
            -webkit-tap-highlight-color: transparent;
            user-select: none;
        }

        #stop-btn:hover {
            transform: scale(1.02);
            box-shadow:
                0 15px 40px rgba(255,0,0,0.3),
                inset 0 -5px 20px rgba(0,0,0,0.3),
                inset 0 5px 20px rgba(255,255,255,0.1);
        }

        #stop-btn:active {
            transform: scale(0.98);
            background: radial-gradient(circle at 30% 30%, #cc3333, #aa0000 50%, #770000);
            box-shadow:
                0 5px 15px rgba(0,0,0,0.5),
                inset 0 5px 20px rgba(0,0,0,0.4);
        }

        #stop-btn span {
            color: white;
            font-size: min(15vw, 80px);
            font-weight: 900;
            text-shadow: 0 2px 4px rgba(0,0,0,0.5);
            letter-spacing: 2px;
        }

        #status {
            margin-top: 30px;
            color: #888;
            font-size: 16px;
            text-align: center;
            min-height: 24px;
        }

        #status.success { color: #4f4; }
        #status.error { color: #f44; }

        .pulse {
            animation: pulse 0.3s ease-out;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255,0,0,0.7); }
            100% { box-shadow: 0 0 0 50px rgba(255,0,0,0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <button id="stop-btn" onclick="sendStop()">
            <span>STOP</span>
        </button>
        <div id="status">Ready</div>
    </div>

    <script>
        let lastClick = 0;

        async function sendStop() {
            // Debounce rapid clicks
            const now = Date.now();
            if (now - lastClick < 200) return;
            lastClick = now;

            const btn = document.getElementById('stop-btn');
            const status = document.getElementById('status');

            // Visual feedback
            btn.classList.remove('pulse');
            void btn.offsetWidth; // Trigger reflow
            btn.classList.add('pulse');

            status.textContent = 'Sending STOP...';
            status.className = '';

            try {
                const response = await fetch('/stop', {
                    method: 'POST',
                    cache: 'no-store'
                });
                const text = await response.text();

                if (response.ok) {
                    status.textContent = text;
                    status.className = 'success';
                } else {
                    status.textContent = 'Error: ' + text;
                    status.className = 'error';
                }
            } catch (e) {
                status.textContent = 'Network error - retrying...';
                status.className = 'error';
                // Auto-retry once
                setTimeout(sendStop, 100);
            }

            // Clear status after 3 seconds
            setTimeout(() => {
                status.textContent = 'Ready';
                status.className = '';
            }, 3000);
        }

        // Also handle spacebar and Enter for keyboard accessibility
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space' || e.code === 'Enter') {
                e.preventDefault();
                sendStop();
            }
        });
    </script>
</body>
</html>
"""

class PanicHandler(BaseHTTPRequestHandler):
    """HTTP request handler for panic stop."""

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_GET(self):
        """Serve the panic button page."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode())

    def do_POST(self):
        """Handle stop command."""
        if self.path == '/stop':
            result = send_stop()
            print(f"[PANIC] STOP triggered: {result}")

            if result.startswith("OK"):
                self.send_response(200)
            else:
                self.send_response(500)

            self.send_header('Content-Type', 'text/plain')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(result.encode())
        else:
            self.send_response(404)
            self.end_headers()

def main():
    """Start the panic stop server."""
    print("\n" + "="*50)
    print("  PANIC STOP SERVER")
    print("="*50)

    # Connect to Arduino
    connect_arduino()

    # Get IP
    ip = get_ip()

    # Start server
    server = HTTPServer(('0.0.0.0', PORT), PanicHandler)

    print(f"\n  Open in browser:")
    print(f"  http://{ip}:{PORT}")
    print(f"\n  Press Ctrl+C to quit")
    print("="*50 + "\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[PANIC] Shutting down...")
        if arduino:
            arduino.close()
        server.shutdown()

if __name__ == "__main__":
    main()
