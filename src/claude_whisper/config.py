import pyaudio
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLAUDE_WHISPER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_name: str = Field(
        default="mlx-community/whisper-large-v3-mlx",
        description="Whisper model name for transcription",
    )

    format: int = Field(
        default=pyaudio.paInt16,
        description="Audio format (16-bit int)",
    )

    channels: int = Field(
        default=1,
        description="Number of audio channels (mono)",
    )

    rate: int = Field(
        default=16000,
        description="Sampling rate in Hz",
    )

    chunk: int = Field(
        default=1024,
        description="Buffer size",
    )

    silence_threshold: int = Field(
        default=500,
        description="Amplitude threshold for detecting silence",
    )

    silence_chunks: int = Field(
        default=30,
        description="Number of consecutive chunks of silence before stopping",
    )

    command: str = Field(
        default="jarvis",
        description="Wake word command to trigger Claude",
    )

    tmux_session_name: str = Field(
        default="claude",
        description="Name of the tmux session",
    )

    permission_mode: str = Field(
        default="acceptEdits",
        description="Permission mode for Claude session (acceptEdits, bypassPermissions, default, delegate, dontAsk, plan)",
    )


config = Config()
