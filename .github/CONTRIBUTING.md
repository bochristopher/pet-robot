# Contributing to Pet Robot

Thanks for your interest in contributing!

## Getting Started

1. Fork the repository
2. Create a feature branch from `tracker`
3. Make your changes
4. Submit a pull request

## Security Requirements

**Before submitting a PR, ensure:**

- [ ] No hardcoded secrets, API keys, or tokens
- [ ] No credentials in comments or documentation
- [ ] User input is validated/sanitized
- [ ] Subprocess calls use list arguments (not shell=True)
- [ ] File paths are validated to prevent traversal

## Code Style

- Python 3.8+ compatible
- Use type hints where practical
- Keep functions focused and small
- Add docstrings for public functions

## Pull Request Process

1. All PRs require at least 1 review
2. Squash merge is used (clean history)
3. Branch is auto-deleted after merge

## Testing

```bash
# Test individual modules
python3 actuators/motor_interface.py --ping
python3 voice/elevenlabs_speaker.py "Hello!"
python3 perception/openai_vision.py --mode brief
```

## Questions?

Open an issue for discussion before starting large changes.
