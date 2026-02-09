# Claude Whisper

Voice-controlled interface for Claude Code using Apple MLX Whisper for speech recognition.

## Overview

Claude Whisper enables voice interaction with Claude Code through push-to-talk functionality. Hold a configurable key (default: ESC), speak your wake word followed by a command, and Claude Whisper will transcribe and execute it using the Claude Agent SDK.

## Features

- Push-to-talk interface with configurable hotkey
- Real-time speech recognition using MLX Whisper (optimized for Apple Silicon)
- Desktop notifications for task status
- Direct integration with Claude Agent SDK
- Configurable wake word to trigger Claude commands
- TOML-based configuration with environment variable overrides

## Requirements

- macOS with Apple Silicon (for MLX acceleration)
- Python 3.10+
- PortAudio (for microphone input)
- Anthropic API key (for Claude Agent SDK)

### macOS Permissions

Your terminal application needs the following permissions enabled in System Settings > Security & Privacy:

- **Input Monitoring** - Required for detecting push-to-talk key presses
- **Accessibility** - Required for keyboard event monitoring

## Installation

### Prerequisites

Install PortAudio (required for microphone input):

```bash
brew install portaudio
```

### Using uvx (Recommended)

```bash
uvx claude-whisper /path/to/your/project
```

### Using pipx

```bash
pipx install claude-whisper
claude-whisper /path/to/your/project
```

### Using pip

```bash
pip install claude-whisper
claude-whisper /path/to/your/project
```

### From Source

1. Install system dependencies:
```bash
make install-deps
```

2. Install the package:
```bash
uv sync
```

## Usage

### Audio Mode (Push-to-Talk)

Run Claude Whisper with a working directory:

```bash
claude-whisper /path/to/your/project
```

Once running:
1. Hold the push-to-talk key (default: ESC)
2. Say your wake word followed by your command (e.g., "Jarvis, create a README for this project")
3. Release the key when done speaking
4. The audio will be transcribed and sent to Claude
5. Desktop notifications will alert you when tasks start and finish

## Configuration

Configure Claude Whisper using environment variables with the `CLAUDE_WHISPER_` prefix or a TOML config file at `~/.config/claude-whisper/config.toml`:

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_WHISPER_MODEL_NAME` | `mlx-community/whisper-medium-mlx-8bit` | Whisper model for transcription |
| `CLAUDE_WHISPER_FORMAT` | `paInt16` | Audio format (16-bit int) |
| `CLAUDE_WHISPER_CHANNELS` | `1` | Number of audio channels (mono) |
| `CLAUDE_WHISPER_RATE` | `16000` | Sampling rate in Hz |
| `CLAUDE_WHISPER_CHUNK` | `1024` | Audio buffer size |
| `CLAUDE_WHISPER_SILENCE_THRESHOLD` | `500` | Amplitude threshold for detecting silence |
| `CLAUDE_WHISPER_SILENCE_CHUNKS` | `30` | Consecutive silent chunks before stopping |
| `CLAUDE_WHISPER_COMMAND` | `jarvis` | Wake word to trigger Claude |
| `CLAUDE_WHISPER_PERMISSION_MODE` | `acceptEdits` | Claude permission mode |
| `CLAUDE_WHISPER_PUSH_TO_TALK_KEY` | `esc` | Key to hold for recording |

### Example TOML Configuration

Create `~/.config/claude-whisper/config.toml`:

```toml
model_name = "mlx-community/whisper-medium-mlx-8bit"
command = "jarvis"
push_to_talk_key = "esc"
permission_mode = "acceptEdits"
```

### Available Push-to-Talk Keys

- `esc`, `escape` - Escape key
- `space` - Space bar
- `enter` - Enter key
- `tab` - Tab key
- `ctrl`, `shift`, `alt`, `cmd` - Modifier keys
- Any single character (e.g., `a`, `z`, `1`)

## Development

### Testing

Run the test suite:
```bash
make test
```

Run tests with coverage:
```bash
pytest --cov=src/claude_whisper --cov-report=term-missing
```

Run specific test files:
```bash
pytest tests/test_config.py
pytest tests/test_main.py
pytest tests/test_integration.py
```

### Code Quality

Format code:
```bash
make format
```

Run linter:
```bash
make lint
```

Fix linting issues:
```bash
make lint-fix
```

Check formatting and linting:
```bash
make check
```

## License

See LICENSE file for details.

## Author

Ashton Sidhu (ashton@sidhulabs.ca)
