import textwrap
import argparse
import asyncio
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4

import mlx_whisper
import numpy as np
import pyaudio
from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, TextBlock, ToolUseBlock
from desktop_notifier import DesktopNotifier
from loguru import logger
from mlx_whisper import load_models
from pynput import keyboard

from .config import config

notifier = DesktopNotifier(app_name="Claude Whisper")
AUDIO_NORMALIZATION_FACTOR = 32768.0


class TaskType(str, Enum):
    """Enumeration of task types."""

    PLAN = "plan"
    EDIT = "edit"


class TransientError(str, Enum):
    API = "API Error"


@dataclass
class TaskContext:
    task_id: str
    task_type: TaskType
    command: str
    working_dir: Path
    permission_mode: str
    started_at: datetime = field(default_factory=datetime.now)

    # Set after completion
    output: str | None = None
    finished_at: datetime | None = None
    error: Exception | None = None


class BaseLifecycle:
    async def on_start(self, ctx: TaskContext):
        message_preview = ctx.command[0:50] + "..."
        await notifier.send(title="Task started", message=f"Started task {message_preview}")

    async def pre_execute(self, ctx: TaskContext):
        pass

    async def execute(self, ctx: TaskContext, client: ClaudeSDKClient):
        pass

    async def on_finish(self, ctx: TaskContext):
        message_preview = ctx.command[0:50] + "..."
        await notifier.send(title="Task Finished", message=f"Finished task {message_preview}")

    async def on_error(self, ctx: TaskContext, error: str):
        await notifier.send(title="Task failed", message=f"Task {ctx.task_id} failed.")
        logger.error(error)

    def is_transient_error(self, block: TextBlock) -> bool:
        return block.text.startswith(TransientError.API)


class EditLifecycle(BaseLifecycle):
    async def execute(self, ctx: TaskContext, client: ClaudeSDKClient):
        await client.query(ctx.command)

        logger.debug("submitted command")

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        if self.is_transient_error(block):
                            await self.on_error(ctx, block.text)
                            await client.interrupt()

                        logger.debug(block.text)


class PlanLifecycle(BaseLifecycle):
    async def pre_execute(self, ctx: TaskContext):
        ctx.command = textwrap.dedent(f"""
            {ctx.command}

            When creating the plan if you are defining classes, include the pseudo code or example implementation.
            DO NOT ask any follow up questions.
        """)

    async def execute(self, ctx: TaskContext, client: ClaudeSDKClient):
        await client.query(ctx.command)

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        if self.is_transient_error(block):
                            await self.on_error(ctx, block.text)
                            await client.interrupt()
                        logger.debug(block.text, end="")
                        print()
                    if isinstance(block, ToolUseBlock) and block.name == "ExitPlanMode":
                        plan: str = block.input.get("plan", "")
                        ctx.output = plan

    async def on_finish(self, ctx: TaskContext):
        message_preview = ctx.command[0:50] + "..."
        if not ctx.output:
            logger.info(f"No plan to save {ctx}")
            await notifier.send(title="Task Finished", message=f"Finished task {message_preview}")
            return

        plan_name = ctx.output.strip().split("\n")[0].removeprefix("# Plan: ").replace(" ", "_").lower()
        logger.info(f"Saving plan to plans/{plan_name}.md")
        with open(f"plans/{plan_name}.md", "w") as f:
            f.write(ctx.output)

        await notifier.send(title="Task Finished", message=f"Finished task {message_preview}")


class TaskTypeDetector:
    # patterns for detecting plan requests
    plan_patterns = [
        r"\bplan\b",
        r"\bdesign\b",
        r"\barchitect(ure)?\b",
        r"\bpropos(e|al)\b",
        r"\bstrategy\b",
        r"\bapproach\s+for\b",
        r"\bblueprint\b",
    ]

    # patterns for detecting edit requests
    edit_patterns = [
        r"\bfix\b",
        r"\bupdate\b",
        r"\bmodify\b",
        r"\bchange\b",
        r"\brefactor\b",
        r"\badd\b.*\bto\b",
        r"\bremove\b",
        r"\bdelete\b",
        r"\bedit\b",
        r"\bimplement\b",
        r"\bcreate\b",
        r"\bwrite\b",
        r"\breplace\b",
        r"\brename\b",
    ]

    def __init__(self):
        self._plan_compiled = [re.compile(p, re.IGNORECASE) for p in self.plan_patterns]
        self._edit_compiled = [re.compile(p, re.IGNORECASE) for p in self.edit_patterns]

    def detect(self, command: str, permission_mode: str | None = None) -> TaskType:
        """
        detect task type from command and permission mode.

        priority:
        1. explicit permission_mode="plan" -> plan
        2. command matches plan patterns -> plan
        3. default -> plan
        """
        # explicit plan mode from permission
        if permission_mode == "plan":
            return TaskType.PLAN

        # pattern matching for plans
        for pattern in self._plan_compiled:
            if pattern.search(command):
                return TaskType.PLAN

        return TaskType.EDIT


class LifecycleManager:
    def __init__(self):
        self.detector = TaskTypeDetector()

    def create_context(self, command: str, working_dir: Path, permission_mode: str) -> TaskContext:
        task_type = self.detector.detect(command, permission_mode)

        return TaskContext(
            task_id=str(uuid4()),
            task_type=task_type,
            command=command,
            working_dir=working_dir,
            permission_mode=permission_mode,
        )

    def get_lifecycle(self, ctx: TaskContext) -> PlanLifecycle | EditLifecycle:
        if ctx.task_type == TaskType.PLAN:
            return PlanLifecycle()

        return EditLifecycle()


lifecycle_manager = LifecycleManager()


class ClaudeSDKSession:
    """Maintains a single conversation session with Claude."""

    def __init__(self, options: ClaudeAgentOptions, ctx: TaskContext):
        self.client = ClaudeSDKClient(options)
        self.ctx = ctx
        self.lifecycle = lifecycle_manager.get_lifecycle(ctx)

    async def run(self, command: str):
        """Execute a command and process the response."""

        try:
            await self.lifecycle.on_start(self.ctx)

            await self.client.connect()

            await self.lifecycle.pre_execute(self.ctx)
            await self.lifecycle.execute(self.ctx, self.client)

            await self.client.disconnect()

            await self.lifecycle.on_finish(self.ctx)
        except Exception as e:
            raise e
            await self.lifecycle.on_error(self.ctx, e)


async def _run_claude_task(command: str, working_dir: Path) -> None:
    """Create and run a Claude task."""

    ctx = lifecycle_manager.create_context(command, working_dir, config.permission_mode)
    permission_mode = config.permission_mode if ctx.task_type == TaskType.EDIT else TaskType.PLAN
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Bash"],
        permission_mode=permission_mode,
        cwd=working_dir,
        system_prompt={"type": "preset", "preset": "claude_code"},
        setting_sources=["project"],
    )
    session = ClaudeSDKSession(options, ctx)

    await session.run(command)


async def _run_bypass_mode(working_dir: Path) -> None:
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


async def _run_audio_mode(working_dir: Path) -> None:
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

            frames = []
            logger.info("Recording...", file=sys.stderr)

            while is_recording.is_set():
                try:
                    stream = audio.open(
                        format=config.format,
                        channels=config.channels,
                        rate=config.rate,
                        input=True,
                        frames_per_buffer=config.chunk,
                    )
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

                # if transcription.startswith(config.command):
                #     claude_command = transcription.removeprefix(config.command).removeprefix(",").strip()
                asyncio.create_task(_run_claude_task(transcription, working_dir))

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        listener.stop()


async def run_whisper(working_dir: Path, bypass_whisper: bool = False) -> None:
    """Main entry point for running Claude Whisper in either bypass or audio mode."""
    working_dir = Path(working_dir).expanduser()

    logger.info(f"Loading model: {config.model_name}")
    load_models.load_model(config.model_name)
    os.makedirs(config.plan_folder, exist_ok=True)

    if bypass_whisper:
        await _run_bypass_mode(working_dir)
    else:
        await _run_audio_mode(working_dir)


def main():
    parser = argparse.ArgumentParser(description="Claude Whisper - Voice-activated Claude AI assistant")
    parser.add_argument("working_dir", type=Path, help="Working directory for Claude sessions")
    parser.add_argument(
        "--bypass-whisper", action="store_true", default=False, help="Bypass whisper and accept text input directly"
    )

    args = parser.parse_args()

    asyncio.run(run_whisper(args.working_dir, args.bypass_whisper))
