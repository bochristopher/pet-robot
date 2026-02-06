# Security Improvements

This document describes the security improvements made to the pet-robot codebase.

## Critical Fixes

### 1. Removed Hardcoded Authentication Token

**Files affected:**
- `setup_env.sh`
- `actuators/motor_interface.py`
- `actuators/simple_motor_server.py`

**Before:** Default token `robot_secret_2024` was hardcoded, allowing anyone to control the robot.

**After:**
- Token MUST be set via `ROBOT_AUTH_TOKEN` environment variable
- Server refuses to start without a token
- Setup script can auto-generate a secure token using `secrets.token_urlsafe(32)`

**To generate a secure token:**
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2. Replaced Pickle with JSON (Unsafe Deserialization)

**File:** `perception/face_recognition_simple.py`

**Before:** Used `pickle.load()` which can execute arbitrary code if the file is tampered with.

**After:** Uses `json.load()` which is safe for data deserialization.

**Note:** If you have existing `.pkl` files, convert them:
```python
import pickle, json
with open('face_labels.pkl', 'rb') as f:
    data = pickle.load(f)
with open('face_labels.json', 'w') as f:
    json.dump({str(k): v for k, v in data.items()}, f)
```

### 3. WebSocket Server Binds to Localhost

**File:** `actuators/simple_motor_server.py`

**Before:** Server bound to `0.0.0.0` (all interfaces), exposing motor control to the network.

**After:**
- Binds to `127.0.0.1` by default (localhost only)
- Set `ROBOT_BIND_HOST=0.0.0.0` only if you need remote access

## Medium Severity Fixes

### 4. Removed API Key Logging

**Files affected:**
- `setup_env.sh`
- `brain/robot_brain.py`
- `voice/whisper_listener.py`
- `voice/elevenlabs_speaker.py`
- `perception/openai_vision.py`

**Before:** Partial API keys were printed to logs (e.g., `key: sk-abc123...`)

**After:** Only prints `[API key configured]` without revealing any key characters.

### 5. Secure Temporary File Handling

**File:** `voice/whisper_listener.py`

**Before:** Used `tempfile.mktemp()` which has race condition vulnerabilities.

**After:** Uses `tempfile.mkstemp()` for atomic file creation.

### 6. Path Traversal Protection

**File:** `recorder.py`

**Before:** User-provided folder path was used directly without validation.

**After:**
- Validates paths are within allowed directories (home, /tmp)
- Blocks writes to sensitive paths (/etc, /usr, /bin, etc.)
- Resolves symlinks to prevent traversal attacks

### 7. Input Sanitization for Speech Synthesis

**Files affected:**
- `smart_explore.py`
- `voice/elevenlabs_speaker.py`

**Before:** User-controlled text passed directly to `subprocess.run()` for espeak.

**After:**
- Text is sanitized (alphanumeric, basic punctuation only)
- Length limited to prevent resource exhaustion
- Timeout added to subprocess calls

## Configuration Requirements

### Required Environment Variables

```bash
# REQUIRED - no defaults for security
export ROBOT_AUTH_TOKEN="your-secure-token-here"

# API Keys (optional but recommended)
export OPENAI_API_KEY="sk-..."
export ELEVENLABS_API_KEY="..."
```

### Optional Security Settings

```bash
# WebSocket server (defaults to localhost)
export ROBOT_BIND_HOST="127.0.0.1"  # or "0.0.0.0" for remote access
export ROBOT_SERVER_PORT="8765"

# Arduino port (auto-detected, but can override)
export ARDUINO_PORT="/dev/ttyACM0"
```

## Remaining Recommendations

1. **Add TLS/SSL** - WebSocket communications should use WSS with certificates
2. **Rate limiting** - Add connection rate limiting to prevent DoS
3. **Logging** - Implement proper logging with rotation for security auditing
4. **Dependency updates** - Regularly update dependencies and scan for CVEs:
   ```bash
   pip install pip-audit
   pip-audit
   ```

## Testing Security Changes

```bash
# Verify token is required
unset ROBOT_AUTH_TOKEN
python3 actuators/simple_motor_server.py  # Should fail

# Set a secure token
export ROBOT_AUTH_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
python3 actuators/simple_motor_server.py  # Should work

# Verify localhost binding
netstat -tlnp | grep 8765  # Should show 127.0.0.1:8765
```
