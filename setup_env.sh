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
    echo "✅ OPENAI_API_KEY already set [REDACTED]"
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
    echo "✅ ELEVENLABS_API_KEY already set [REDACTED]"
fi

# Robot server settings (defaults)
export ROBOT_SERVER_HOST="${ROBOT_SERVER_HOST:-localhost}"
export ROBOT_SERVER_PORT="${ROBOT_SERVER_PORT:-8765}"

# Authentication token - REQUIRED, no default for security
if [ -z "$ROBOT_AUTH_TOKEN" ]; then
    echo ""
    echo "⚠️  ROBOT_AUTH_TOKEN not set"
    echo "   Generate a secure token with: python3 -c \"import secrets; print(secrets.token_urlsafe(32))\""
    echo ""
    read -p "Enter your auth token (or press Enter to generate one): " token
    if [ -n "$token" ]; then
        export ROBOT_AUTH_TOKEN="$token"
    else
        export ROBOT_AUTH_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
        echo "   Generated token: $ROBOT_AUTH_TOKEN"
        echo "   ⚠️  Save this token - you'll need it for client connections!"
    fi
    echo "✅ ROBOT_AUTH_TOKEN set"
else
    echo "✅ ROBOT_AUTH_TOKEN already set [REDACTED]"
fi

echo ""
echo "📡 Server: ws://$ROBOT_SERVER_HOST:$ROBOT_SERVER_PORT"
echo ""
echo "================================"
echo "✅ Environment ready!"
echo ""
echo "Run: python3 robot_pet.py"
echo "Or:  python3 test_modules.py"


