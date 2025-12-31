#!/bin/bash
# Robot Pet Environment Setup
# Source this file before running robot_pet.py:
#   source setup_env.sh

echo "🤖 Robot Pet Environment Setup"
echo "================================"

# Check/prompt for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "⚠️  OPENAI_API_KEY not set"
    echo "   Get your key from: https://platform.openai.com/api-keys"
    echo ""
    read -p "Enter your OpenAI API key (or press Enter to skip): " key
    if [ -n "$key" ]; then
        export OPENAI_API_KEY="$key"
        echo "✅ OPENAI_API_KEY set"
    else
        echo "⚠️  Skipped - Vision and Brain will not work"
    fi
else
    echo "✅ OPENAI_API_KEY already set: ${OPENAI_API_KEY:0:8}..."
fi

# Check/prompt for ElevenLabs API key
if [ -z "$ELEVENLABS_API_KEY" ]; then
    echo ""
    echo "⚠️  ELEVENLABS_API_KEY not set"
    echo "   Get your key from: https://elevenlabs.io/api"
    echo "   (Optional - will use fallback TTS if not set)"
    echo ""
    read -p "Enter your ElevenLabs API key (or press Enter to skip): " key
    if [ -n "$key" ]; then
        export ELEVENLABS_API_KEY="$key"
        echo "✅ ELEVENLABS_API_KEY set"
    else
        echo "⚠️  Skipped - Will use pyttsx3/espeak fallback"
    fi
else
    echo "✅ ELEVENLABS_API_KEY already set: ${ELEVENLABS_API_KEY:0:8}..."
fi

# Robot server settings (defaults)
export ROBOT_SERVER_HOST="${ROBOT_SERVER_HOST:-localhost}"
export ROBOT_SERVER_PORT="${ROBOT_SERVER_PORT:-8765}"
export ROBOT_AUTH_TOKEN="${ROBOT_AUTH_TOKEN:-robot_secret_2024}"

echo ""
echo "📡 Server: ws://$ROBOT_SERVER_HOST:$ROBOT_SERVER_PORT"
echo ""
echo "================================"
echo "✅ Environment ready!"
echo ""
echo "Run: python3 robot_pet.py"
echo "Or:  python3 test_modules.py"


