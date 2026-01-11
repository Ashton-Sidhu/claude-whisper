import argparse
import asyncio
import os
import signal
import subprocess
import sys

import mlx_whisper
import numpy as np
import pyaudio
from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, TextBlock
from desktop_notifier import DesktopNotifier, ReplyField
from loguru import logger

from .config import config

notifier = DesktopNotifier(app_name="Claude Whisper")


class ClaudeSDKSession:
    """Maintains a single conversation session with Claude."""

    def __init__(self, options: ClaudeAgentOptions | None = None):
        self.client = ClaudeSDKClient(options)
        self.turn_count = 0

    async def run(self, command: str):
        logger.info(f"Received command: {command}")
        await self.client.connect()
        await self.client.query(command)
        message = command[0:50] + "..."
        await notifier.send(
            title="Task started",
            message=f"Started task {message}",
        )

        # Process response
        async for message in self.client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text, end="")
        print()  # New line after response

        await notifier.send(
            title="Task Finished",
            message=f"Finished task {message}",
            # reply_field=ReplyField(
            #     on_replied=lambda text: print(text),
            # ),
        )
        await self.client.disconnect()


# class ClaudeTmux:
#     def __init__(self, claude_working_dir: str):
#         self.claude_working_dir = claude_working_dir

#     @property
#     def session_name(self) -> str:
#         return config.tmux_session_name

#     @property
#     def num_windows(self) -> int:
#         proc = subprocess.Popen(
#             ["tmux", "display-message", "-t", self.session_name, "-p", "'#{session_windows}'"],
#             stdout=subprocess.PIPE,
#             text=True,
#         )

#         if proc.returncode:
#             logger.error("Error getting number of windows from tmux session")
#             return -1

#         if proc.stdout:
#             output = proc.stdout.read().strip().replace("'", "")

#             if not output:
#                 return 0

#             return int(output)

#         return 0

#     def session_exists(self) -> bool:
#         proc = subprocess.Popen(["tmux", "has-session", "-t", f"={self.session_name}"])

#         proc.wait()

#         # 1 means session does not exist, 0 means it does, so we flip the bool with not
#         return not bool(proc.returncode)

#     def run(self, command):
#         window_name = self.num_windows + 1

#         # Build the claude command with permission mode
#         claude_cmd = f"claude --permission-mode {config.permission_mode} '{command}'"

#         if not self.session_exists():
#             logger.info("Creating new session and running claude.")
#             # -c start dir, -n window name
#             proc = subprocess.Popen(
#                 [
#                     "tmux",
#                     "new",
#                     "-s",
#                     self.session_name,
#                     "-d",
#                     "-n",
#                     str(window_name),
#                     "-c",
#                     self.claude_working_dir,
#                     claude_cmd,
#                 ],
#                 stdout=subprocess.DEVNULL,
#             )

#         else:
#             logger.info("Creating new window and running claude.")
#             proc = subprocess.Popen(
#                 [
#                     "tmux",
#                     "neww",
#                     "-t",
#                     f"{self.session_name}:{window_name}",
#                     "-d",
#                     "-c",
#                     self.claude_working_dir,
#                     claude_cmd,
#                 ],
#                 stdout=subprocess.DEVNULL,
#             )

#         if proc.returncode:
# logger.error("Error creating claude tmux session.")

#         return ClaudeSession(session_name=self.session_name, window=window_name, inputs=[command])


ACTIVE_SESSIONS: list[ClaudeSDKSession] = []


async def run_whisper(working_dir: str, bypass_whisper: bool = False) -> None:
    claude_working_dir = os.path.expanduser(working_dir)
    # tmux = ClaudeTmux(claude_working_dir)

    # Bypass mode: accept text input directly
    if bypass_whisper:
        logger.info("Running in bypass mode - accepting text input directly")
        while True:
            try:
                print("Enter command: ", end="", file=sys.stderr)
                # Use to_thread to make input non-blocking so background tasks can run
                user_input = await asyncio.to_thread(input)
                user_input = user_input.strip()

                if not user_input:
                    continue

                options = ClaudeAgentOptions(
                    allowed_tools=["Read", "Write", "Bash"],
                    permission_mode="acceptEdits",
                    cwd=claude_working_dir,
                    system_prompt={
                        "type": "preset",
                        "preset": "claude_code",  # Use Claude Code's system prompt
                    },
                    setting_sources=["project"],
                )
                session = ClaudeSDKSession(options)

                # Start the session in the background without waiting
                asyncio.create_task(session.run(user_input))

                # session = tmux.run(user_input)
                ACTIVE_SESSIONS.append(session)

            except (KeyboardInterrupt, EOFError):
                logger.info("Exiting bypass mode")
                break
        return

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
            data = await asyncio.to_thread(stream.read, config.chunk, exception_on_overflow=False)
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
                options = ClaudeAgentOptions(
                    allowed_tools=["Read", "Write", "Bash"],
                    permission_mode="acceptEdits",
                    cwd=claude_working_dir,
                    system_prompt={
                        "type": "preset",
                        "preset": "claude_code",  # Use Claude Code's system prompt
                    },
                    setting_sources=["project"],
                )
                session = ClaudeSDKSession(options)

                # Start the session in the background without waiting
                asyncio.create_task(session.run(claude_command))

                # session = tmux.run(user_input)
                ACTIVE_SESSIONS.append(session)


def main():
    parser = argparse.ArgumentParser(description="Claude Whisper - Voice-activated Claude AI assistant")
    parser.add_argument("working_dir", type=str, help="Working directory for Claude sessions")
    parser.add_argument(
        "--bypass-whisper", action="store_true", default=False, help="Bypass whisper and accept text input directly"
    )

    args = parser.parse_args()

    asyncio.run(run_whisper(args.working_dir, args.bypass_whisper))
