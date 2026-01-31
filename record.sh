#!/usr/bin/env bash
DIR=~/data/$(date +%Y%m%d_%H%M%S)
mkdir -p "$DIR"
echo "Recording to $DIR"
python3 "$(dirname "$0")/recorder.py" "$DIR"
