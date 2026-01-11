import argparse
import asyncio
import os
import sys

import mlx_whisper
import numpy as np
import pyaudio
from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, TextBlock
from desktop_notifier import DesktopNotifier
from loguru import logger
from mlx_whisper import load_models
from pynput import keyboard

from .config import config

notifier = DesktopNotifier(app_name="Claude Whisper")
AUDIO_NORMALIZATION_FACTOR = 32768.0


class ClaudeSDKSession:
    """Maintains a single conversation session with Claude."""

    def __init__(self, options: ClaudeAgentOptions):
        self.client = ClaudeSDKClient(options)

    async def run(self, command: str):
        """Execute a command and process the response."""
        logger.debug(f"Received command: {command}")

        await self.client.connect()
        await self.client.query(command)

        message_preview = command[0:50] + "..."
        await notifier.send(title="Task started", message=f"Started task {message_preview}")

        async for message in self.client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        logger.debug(block.text, end="")

        print()

        await notifier.send(title="Task Finished", message=f"Finished task {message_preview}")
        await self.client.disconnect()


async def _run_claude_task(command: str, working_dir: str) -> None:
    """Create and run a Claude task."""
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Bash"],
        permission_mode="acceptEdits",
        cwd=working_dir,
        system_prompt={"type": "preset", "preset": "claude_code"},
        setting_sources=["project"],
    )
    session = ClaudeSDKSession(options)
    await session.run(command)


async def _run_bypass_mode(working_dir: str) -> None:
    """Run in bypass mode - accept text input directly."""
    logger.info("Running in bypass mode - accepting text input directly")
    while True:
        try:
            print("Enter command: ", end="", file=sys.stderr)
            user_input = await asyncio.to_thread(input)
            user_input = user_input.strip()

            if user_input:
                asyncio.create_task(_run_claude_task(user_input, working_dir))

        except (KeyboardInterrupt, EOFError):
            logger.info("Exiting bypass mode")
            break


def _parse_push_to_talk_key(key_string: str):
    """Parse a single key name into a pynput Key or KeyCode object."""
    key_string = key_string.strip().lower()

    key_map = {
        "esc": keyboard.Key.esc,
        "escape": keyboard.Key.esc,
        "space": keyboard.Key.space,
        "enter": keyboard.Key.enter,
        "tab": keyboard.Key.tab,
        "ctrl": keyboard.Key.ctrl_l,
        "shift": keyboard.Key.shift_l,
        "alt": keyboard.Key.alt_l,
        "cmd": keyboard.Key.cmd_l,
    }

    if key_string in key_map:
        return key_map[key_string]
    elif len(key_string) == 1:
        return keyboard.KeyCode.from_char(key_string)
    else:
        logger.warning(f"Unknown key: {key_string}, defaulting to ESC")
        return keyboard.Key.esc


async def _run_audio_mode(working_dir: str) -> None:
    """Run in audio mode - push-to-talk with configurable key."""
    push_to_talk_key = _parse_push_to_talk_key(config.push_to_talk_key)
    logger.info(f"Push-to-talk key: {config.push_to_talk_key}")

    loop = asyncio.get_running_loop()
    is_recording = asyncio.Event()

    def on_press(key):
        """Start recording when push-to-talk key is pressed."""
        if key == push_to_talk_key:
            logger.debug("Push-to-talk activated")
            loop.call_soon_threadsafe(is_recording.set)

    def on_release(key):
        """Stop recording when push-to-talk key is released."""
        if key == push_to_talk_key:
            logger.debug("Push-to-talk released")
            loop.call_soon_threadsafe(is_recording.clear)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    logger.info(f"Keyboard listener started. Hold '{config.push_to_talk_key}' to record...")

    audio = pyaudio.PyAudio()
    try:
        while True:
            await is_recording.wait()

            stream = audio.open(
                format=config.format,
                channels=config.channels,
                rate=config.rate,
                input=True,
                frames_per_buffer=config.chunk,
            )

            frames = []
            logger.info("Recording...", file=sys.stderr)

            while is_recording.is_set():
                try:
                    data = await asyncio.to_thread(stream.read, config.chunk, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    frames.append(audio_data)
                except Exception as e:
                    logger.error(f"Error reading audio: {e}")
                    break

            logger.info("Stopped recording", file=sys.stderr)

            if frames:
                audio_data = np.concatenate(frames)

                if np.max(np.abs(audio_data)) < config.silence_threshold:
                    logger.info("No audio detected")
                    continue

                audio_data = audio_data.astype(np.float32) / AUDIO_NORMALIZATION_FACTOR
                logger.info("Transcribing audio...")
                result = await asyncio.to_thread(
                    mlx_whisper.transcribe,
                    audio_data,
                    path_or_hf_repo=config.model_name,
                    language="en",
                    prompt="{config.command}, linting",
                )
                transcription = result["text"].strip().lower()
                logger.info(f"Transcription: {transcription}")

                if transcription.startswith(config.command):
                    claude_command = transcription.removeprefix(config.command).removeprefix(",").strip()
                    asyncio.create_task(_run_claude_task(claude_command, working_dir))

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        listener.stop()


async def run_whisper(working_dir: str, bypass_whisper: bool = False) -> None:
    """Main entry point for running Claude Whisper in either bypass or audio mode."""
    working_dir = os.path.expanduser(working_dir)

    logger.info(f"Loading model: {config.model_name}")
    load_models.load_model(config.model_name)

    if bypass_whisper:
        await _run_bypass_mode(working_dir)
    else:
        await _run_audio_mode(working_dir)


def main():
    parser = argparse.ArgumentParser(description="Claude Whisper - Voice-activated Claude AI assistant")
    parser.add_argument("working_dir", type=str, help="Working directory for Claude sessions")
    parser.add_argument(
        "--bypass-whisper", action="store_true", default=False, help="Bypass whisper and accept text input directly"
    )

    args = parser.parse_args()

    asyncio.run(run_whisper(args.working_dir, args.bypass_whisper))
