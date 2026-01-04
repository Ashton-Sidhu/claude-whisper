# Claude Whisper

Voice-controlled interface for Claude Code using Apple MLX Whisper for speech recognition.

## Overview

Claude Whisper enables hands-free interaction with Claude Code by listening for voice commands and automatically executing them in tmux sessions. Simply say your wake word followed by a command, and Claude Whisper will transcribe and execute it.

## Features

- Real-time speech recognition using MLX Whisper (optimized for Apple Silicon)
- Automatic silence detection to identify when you've stopped speaking
- Configurable wake word to trigger Claude commands
- Integration with tmux to manage multiple Claude sessions
- Customizable permission modes for Claude Code
- Environment-based configuration

## Requirements

- macOS with Apple Silicon (for MLX acceleration)
- Python 3.10+
- tmux
- PortAudio
- Claude Code CLI

## Installation

1. Install system dependencies:
```bash
make install-deps
```

2. Install the package:
```bash
uv sync
```

## Usage

Run Claude Whisper with a working directory for Claude Code:

```bash
claude-whisper /path/to/your/project
```

Once running:
1. The application will listen continuously for audio input
2. Say your wake word (default: "jarvis")
3. Follow with your command for Claude
4. The command will be transcribed and executed in a new tmux window

### Example

```
"Jarvis, create a README for this project"
```

This will create a new tmux window running Claude Code with the prompt "create a README for this project".

## Configuration

Configure Claude Whisper using environment variables with the `CLAUDE_WHISPER_` prefix or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_WHISPER_MODEL_NAME` | `mlx-community/whisper-large-v3-mlx` | Whisper model for transcription |
| `CLAUDE_WHISPER_FORMAT` | `paInt16` | Audio format (16-bit int) |
| `CLAUDE_WHISPER_CHANNELS` | `1` | Number of audio channels (mono) |
| `CLAUDE_WHISPER_RATE` | `16000` | Sampling rate in Hz |
| `CLAUDE_WHISPER_CHUNK` | `1024` | Audio buffer size |
| `CLAUDE_WHISPER_SILENCE_THRESHOLD` | `500` | Amplitude threshold for detecting silence |
| `CLAUDE_WHISPER_SILENCE_CHUNKS` | `30` | Consecutive silent chunks before stopping |
| `CLAUDE_WHISPER_COMMAND` | `jarvis` | Wake word to trigger Claude |
| `CLAUDE_WHISPER_TMUX_SESSION_NAME` | `claude` | Name of the tmux session |
| `CLAUDE_WHISPER_PERMISSION_MODE` | `acceptEdits` | Claude permission mode |

### Permission Modes

- `acceptEdits`: Automatically accept file edits
- `bypassPermissions`: Bypass all permission prompts
- `default`: Use Claude's default permission handling
- `delegate`: Delegate permission decisions
- `dontAsk`: Don't ask for permissions
- `plan`: Enter planning mode

## Development

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

## How It Works

1. **Audio Capture**: Uses PyAudio to continuously capture audio from your microphone
2. **Silence Detection**: Monitors audio amplitude to detect when you start and stop speaking
3. **Transcription**: Processes audio chunks with MLX Whisper for fast, on-device transcription
4. **Wake Word Detection**: Checks if transcription starts with the configured wake word
5. **Command Execution**: Strips the wake word and sends the remaining text to Claude Code in a new tmux window

## Project Structure

```
claude-whisper/
├── src/
│   └── claude_whisper/
│       ├── __init__.py      # Main application logic
│       └── config.py        # Configuration management
├── pyproject.toml           # Project metadata and dependencies
├── Makefile                 # Development commands
└── README.md                # This file
```

## License

See LICENSE file for details.

## Author

Ashton Sidhu (ashton.sidhu1994@gmail.com)
