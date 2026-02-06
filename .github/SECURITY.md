# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

1. **Do NOT open a public issue** for security vulnerabilities
2. Email the maintainer directly or use GitHub's private vulnerability reporting
3. Include details about the vulnerability and steps to reproduce

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| tracker | :white_check_mark: |
| main    | :white_check_mark: |

## Security Best Practices

When using this robot:

1. **Never commit secrets** - Use environment variables for API keys and tokens
2. **Use strong auth tokens** - Generate with `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`
3. **Keep dependencies updated** - Run `pip-audit` regularly
4. **Bind to localhost** - Only expose motor server to network if necessary

## Known Security Considerations

- Motor control server should only be exposed on trusted networks
- API keys (OpenAI, ElevenLabs) should never be committed
- Face recognition data is stored locally - protect the data directory
