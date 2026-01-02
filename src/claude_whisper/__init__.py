from typing import Literal
import pyaudio
import numpy as np
import sys
import mlx_whisper
from loguru import logger
import subprocess

MODEL_NAME = "mlx-community/whisper-large-v3-mlx"  # Smaller model for faster processing

# PyAudio configuration
FORMAT = pyaudio.paInt16  # Audio format (16-bit int)
CHANNELS = 1  # Number of audio channels (mono)
RATE = 16000  # Sampling rate (16 kHz)
CHUNK = 1024  # Buffer size
SILENCE_THRESHOLD = 500  # Amplitude threshold for detecting silence
SILENCE_CHUNKS = 30  # Number of consecutive chunks of silence before stopping

COMMAND = "jarvis"


class Tmux:
    @property
    def session_name(self) -> Literal["claude"]:
        return "claude"

    def session_exists(self) -> bool:
        proc = subprocess.Popen(
            ["tmux", "has-session", "-t", self.session_name],
        )

        proc.wait()

        return not bool(proc.returncode)

    def run(self, command):
        if not self.session_exists():
            logger.info("Creating new session and running claude.")
            proc = subprocess.Popen(
                ["tmux", "new", "-s", self.session_name, "-d", f"claude '{command}'"],
                stdout=subprocess.DEVNULL,
            )

        else:
            logger.info("Creating new window and running claude.")
            proc = subprocess.Popen(
                ["tmux", "neww", "-t", self.session_name, "-d", f"claude '{command}'"],
                stdout=subprocess.DEVNULL,
            )

        if proc.returncode:
            logger.error("Error creating claude tmux session.")

        return proc.returncode


def main() -> None:
    tmux = Tmux()

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    while True:
        print("Listening...", file=sys.stderr)

        frames = []  # List to store audio chunks
        silent_chunks = 0

        print("Waiting for speech...", file=sys.stderr)

        # Listen until we detect speech
        while True:
            # Read audio data from the microphone
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Check if audio_data exceeds the silence threshold
            if np.max(np.abs(audio_data)) < SILENCE_THRESHOLD:
                silent_chunks += 1
            else:
                silent_chunks = 0

            # If we have enough silence chunks, consider it the end of the speech
            if silent_chunks > SILENCE_CHUNKS:
                break

            # Accumulate frames if we detect sound above the threshold
            frames.append(audio_data)

        # Concatenate all audio data in frames for a single transcription
        if frames:
            audio_data = np.concatenate(frames)

            max_amplitude = np.max(np.abs(audio_data))

            # logger.debug("Max amplitude", max_amplitude)

            if max_amplitude < SILENCE_THRESHOLD:
                logger.info("No audio")
                continue

            audio_data = audio_data.astype(np.float32) / 32768.0

            logger.info("Sending audio data to model")
            # Process audio with mlx_whisper
            result = mlx_whisper.transcribe(audio_data, path_or_hf_repo=MODEL_NAME, language="en")
            transcription = result["text"].strip().lower()  # Normalize text for comparison

            logger.info(transcription)

            # Output to stdout for piping
            if transcription.startswith(COMMAND):
                # logger.debug("running command: ", COMMAND)
                claude_command = transcription.removeprefix(COMMAND).removeprefix(",").strip()

                tmux.run(claude_command)
