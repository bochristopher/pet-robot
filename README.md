# 🤖 Robot Pet - Spark

An autonomous robot pet built on **Jetson Orin Nano** with voice interaction, computer vision, and intelligent exploration.

## ✅ Working Features

- **Voice Interaction** - Wake word ("hey robot") with Vosk STT
- **Natural Speech** - ElevenLabs TTS with emotion and caching
- **Computer Vision** - GPT-4V scene understanding
- **Motor Control** - 4-wheel omnidirectional movement
- **AI Personality** - GPT-4 conversational brain
- **1-Minute Conversation** - Stays awake without repeating wake word

## 🛠️ Hardware

- Jetson Orin Nano (8GB RAM, CUDA 12.6)
- Arduino Mega 2560 (4 omnidirectional wheels)
- USB Camera (640x480)
- USB Microphone
- USB Speaker

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Setup API Keys
```bash
# Get API keys from:
# - ElevenLabs: https://elevenlabs.io/
# - OpenAI: https://platform.openai.com/

export ELEVENLABS_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"

# Or add to ~/.bashrc for persistence
echo 'export ELEVENLABS_API_KEY="your_key"' >> ~/.bashrc
echo 'export OPENAI_API_KEY="your_key"' >> ~/.bashrc
```

### 3. Download Vosk Speech Model
```bash
mkdir -p ~/ml_models && cd ~/ml_models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

### 4. Run Voice Test
```bash
python3 /tmp/voice_test.py
```

Say **"hey robot"** then ask questions!

## 🎮 Usage

### Voice Commands
```
"hey robot"           → Wake up
"what do you see?"    → Computer vision
"how are you?"        → Chat
"tell me a joke"      → Entertainment
```

Vision triggers: "what you see", "look around", "describe", "can you see"

### Test Individual Modules
```bash
# Speaker
python3 elevenlabs_speaker.py "Hello!"

# Vision
python3 openai_vision.py --mode brief

# Voice
python3 voice_listener.py

# Motors
python3 motor_interface.py --ping
```

## 📁 Project Structure

```
robot_pet/
├── robot_pet.py           # Main integration
├── voice_listener.py      # Vosk STT
├── elevenlabs_speaker.py  # ElevenLabs TTS
├── openai_vision.py       # GPT-4V vision
├── motor_interface.py     # Motor control
├── robot_brain.py         # GPT-4 personality
└── requirements.txt
```

## 💰 Cost Tracking

- **ElevenLabs TTS**: ~$0.001/phrase (cached)
- **OpenAI Vision**: ~$0.01/image
- **OpenAI Chat**: ~$0.03/1000 tokens

All modules track usage: `module.get_stats()`

## 🔧 Configuration

**Volume** (elevenlabs_speaker.py:114):
```python
"volume=0.4"  # 0.0-1.0
```

**Wake Words** (voice_listener.py:58):
```python
WAKE_WORDS = ["hey robot", "robot"]
```

**Conversation Timeout** (voice_listener.py:55):
```python
CONVERSATION_TIMEOUT = 60.0  # seconds
```

## 🐛 Troubleshooting

**Poor recognition?** Download larger model:
```bash
bash /tmp/upgrade_vosk.sh  # 1.8GB model
```

**No audio?** Check device:
```bash
aplay -L
```

## 📄 License

MIT

## 👤 Author

Christopher Bo - [@bochristopher](https://github.com/bochristopher)
