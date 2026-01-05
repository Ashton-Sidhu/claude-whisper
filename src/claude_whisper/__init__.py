import os
import subprocess
import sys

import mlx_whisper
import numpy as np
import pyaudio
import typer
from loguru import logger

from .config import config

app = typer.Typer()


class ClaudeSession:
    def __init__(self, session_name: str, window: int, inputs: list[str]):
        self.session_name = session_name
        self.window = window
        self.inputs = inputs


class ClaudeTmux:
    def __init__(self, claude_working_dir: str):
        self.claude_working_dir = claude_working_dir

    @property
    def session_name(self) -> str:
        return config.tmux_session_name

    @property
    def num_windows(self) -> int:
        proc = subprocess.Popen(
            ["tmux", "display-message", "-t", self.session_name, "-p", "'#{session_windows}'"],
            stdout=subprocess.PIPE,
            text=True,
        )

        if proc.returncode:
            logger.error("Error getting number of windows from tmux session")
            return -1

        if proc.stdout:
            output = proc.stdout.read().strip().replace("'", "")

            if not output:
                return 0

            return int(output)

        return 0

    def session_exists(self) -> bool:
        proc = subprocess.Popen(["tmux", "has-session", "-t", f"={self.session_name}"])

        proc.wait()

        # 1 means session does not exist, 0 means it does, so we flip the bool with not
        return not bool(proc.returncode)

    def run(self, command):
        window_name = self.num_windows + 1

        # Build the claude command with permission mode
        claude_cmd = f"claude --permission-mode {config.permission_mode} '{command}'"

        if not self.session_exists():
            logger.info("Creating new session and running claude.")
            # -c start dir, -n window name
            proc = subprocess.Popen(
                [
                    "tmux",
                    "new",
                    "-s",
                    self.session_name,
                    "-d",
                    "-n",
                    str(window_name),
                    "-c",
                    self.claude_working_dir,
                    claude_cmd,
                ],
                stdout=subprocess.DEVNULL,
            )

        else:
            logger.info("Creating new window and running claude.")
            proc = subprocess.Popen(
                [
                    "tmux",
                    "neww",
                    "-t",
                    f"{self.session_name}:{window_name}",
                    "-d",
                    "-c",
                    self.claude_working_dir,
                    claude_cmd,
                ],
                stdout=subprocess.DEVNULL,
            )

        if proc.returncode:
            logger.error("Error creating claude tmux session.")

        return ClaudeSession(session_name=self.session_name, window=window_name, inputs=[command])


ACTIVE_SESSIONS: list[ClaudeSession] = []


@app.command()
def main(working_dir: str) -> None:
    claude_working_dir = os.path.expanduser(working_dir)
    tmux = ClaudeTmux(claude_working_dir)

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=config.format,
        channels=config.channels,
        rate=config.rate,
        input=True,
        frames_per_buffer=config.chunk,
    )

    while True:
        print("Listening...", file=sys.stderr)

        frames = []  # List to store audio chunks
        silent_chunks = 0

        print("Waiting for speech...", file=sys.stderr)

        # Listen until we detect speech
        while True:
            # Read audio data from the microphone
            data = stream.read(config.chunk, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Check if audio_data exceeds the silence threshold
            if np.max(np.abs(audio_data)) < config.silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0

            # If we have enough silence chunks, consider it the end of the speech
            if silent_chunks > config.silence_chunks:
                break

            # Accumulate frames if we detect sound above the threshold
            frames.append(audio_data)

        # Concatenate all audio data in frames for a single transcription
        if frames:
            audio_data = np.concatenate(frames)

            max_amplitude = np.max(np.abs(audio_data))

            # logger.debug("Max amplitude", max_amplitude)

            if max_amplitude < config.silence_threshold:
                logger.info("No audio")
                continue

            audio_data = audio_data.astype(np.float32) / 32768.0

            logger.info("Sending audio data to model")
            # Process audio with mlx_whisper
            result = mlx_whisper.transcribe(
                audio_data, path_or_hf_repo=config.model_name, language="en", prompt=config.command
            )
            transcription = result["text"].strip().lower()  # Normalize text for comparison

            logger.info(transcription)

            # Output to stdout for piping
            if transcription.startswith(config.command):
                # logger.debug("running command: ", config.command)
                claude_command = transcription.removeprefix(config.command).removeprefix(",").strip()

                logger.debug(claude_command)
                session = tmux.run(claude_command)
                ACTIVE_SESSIONS.append(session)


app()
