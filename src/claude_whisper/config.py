import os
from typing import Literal

from pathlib import Path

import pyaudio
from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, TomlConfigSettingsSource

PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLAUDE_WHISPER_",
        toml_file=os.path.expanduser("~/.config/claude-whisper/config.toml"),
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    model_name: str = Field(
        default="mlx-community/whisper-medium-mlx-8bit",
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
        description="Wake word command to trigger Claude. This does nothing right now and will probably be deleted.",
    )

    permission_mode: PermissionMode = Field(
        default="acceptEdits",
        description=(
            "Permission mode for Claude session (acceptEdits, bypassPermissions, default, plan)"
        ),
    )

    push_to_talk_key: str = Field(
        default="esc",
        description="Key or key combination for push-to-talk (e.g., 'esc', 'ctrl+shift+r')",
    )

    plan_folder: str = Field(
        default="plans",
        description="Folder inside the repository where Claudes design plans should go.",
    )

    cwd: Path = Field(
        default_factory=Path.cwd,
        description="Working directory for Claude sessions. Defaults to the current working directory.",
    )


config = Config()
