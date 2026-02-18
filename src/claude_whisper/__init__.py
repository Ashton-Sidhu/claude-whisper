import argparse
import asyncio
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4

import numpy as np
from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, TextBlock, ToolUseBlock
from loguru import logger

from .config import config
from .data import DEV_VOCABULARY, EDIT_PATTERNS, NOISE_PHRASES, PLAN_PATTERNS, WORD_CORRECTIONS
from .tools import claude_whisper_mcp_server

_import_start = time.perf_counter()

logger.info("Loading MLX whisper, if this is the first time, it may take a while - hang tight!")
_t = time.perf_counter()
import mlx_whisper

print(f"[import timing] mlx_whisper: {time.perf_counter() - _t:.3f}s", file=sys.stderr)

try:
    _t = time.perf_counter()
    import pyaudio

    print(f"[import timing] pyaudio: {time.perf_counter() - _t:.3f}s", file=sys.stderr)
except ImportError as e:
    if "portaudio" in str(e).lower() or "could not import" in str(e).lower():
        print(
            "\nError: PortAudio library not found.\n"
            "PyAudio requires PortAudio to be installed.\n\n"
            "Install it with Homebrew:\n"
            "    brew install portaudio\n\n"
            "Then reinstall pyaudio:\n"
            "    pip install --force-reinstall pyaudio\n",
            file=sys.stderr,
        )
        sys.exit(1)
    raise


_t = time.perf_counter()
from desktop_notifier import DesktopNotifier

print(f"[import timing] desktop_notifier: {time.perf_counter() - _t:.3f}s", file=sys.stderr)

from mlx_whisper import load_models

_t = time.perf_counter()
from pynput import keyboard

print(f"[import timing] pynput: {time.perf_counter() - _t:.3f}s", file=sys.stderr)


print(f"[import timing] total imports: {time.perf_counter() - _import_start:.3f}s", file=sys.stderr)

notifier = DesktopNotifier(app_name="Claude Whisper")
AUDIO_NORMALIZATION_FACTOR = 32768.0


def apply_word_corrections(text: str, corrections: dict[str, str] | None = None) -> str:
    """Apply word corrections to transcribed text.

    Performs case-insensitive replacement of known misinterpretations.
    Longer patterns are matched first to avoid partial replacements.
    """
    if corrections is None:
        corrections = WORD_CORRECTIONS

    # Sort by length descending so longer phrases match first
    sorted_corrections = sorted(corrections.items(), key=lambda x: len(x[0]), reverse=True)

    result = text
    for wrong, correct in sorted_corrections:
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        result = pattern.sub(correct, result)

    return result


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
    async def pre_execute(self, ctx: TaskContext):
        ctx.command = textwrap.dedent(f"""
            {ctx.command}

            DO NOT ask any follow up questions.
            DO NOT ask for any clarification. Always go with your recommended approach and state your assumptions at the end.
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
    plan_patterns = PLAN_PATTERNS
    edit_patterns = EDIT_PATTERNS

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


async def _run_claude_task(command: str) -> None:
    """Create and run a Claude task."""
    working_dir = config.cwd

    ctx = lifecycle_manager.create_context(command, working_dir, config.permission_mode)

    logger.debug(f"Starting task with context: {ctx}")

    permission_mode = config.permission_mode if ctx.task_type == TaskType.EDIT else TaskType.PLAN
    options = ClaudeAgentOptions(
        allowed_tools=["mcp__utils__screenshot"],
        disallowed_tools=["AskUserQuestion"],
        permission_mode=permission_mode,
        model=config.model,
        cwd=working_dir,
        system_prompt={"type": "preset", "preset": "claude_code"},
        setting_sources=["project"],
        mcp_servers={"utils": claude_whisper_mcp_server},
        max_buffer_size=5 * 1024 * 1024,
        env={
            "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
            "OTEL_LOGS_EXPORTER": "otlp",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
            "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT": os.getenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", ""),
            "OTEL_EXPORTER_OTLP_HEADERS": os.getenv("OTEL_EXPORTER_OTLP_HEADERS", ""),
            "OTEL_LOG_USER_PROMPTS": "1",
            "OTEL_LOG_TOOL_DETAILS": "1",
        },
    )
    session = ClaudeSDKSession(options, ctx)

    await session.run(command)


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


async def _run_audio_mode() -> None:
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

            if is_recording.is_set():
                await notifier.send(title="Claude Whisper", message="Listening")

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

                whisper_prompt = ", ".join([config.command] + DEV_VOCABULARY)
                result = await asyncio.to_thread(
                    mlx_whisper.transcribe,
                    audio_data,
                    path_or_hf_repo=config.model_name,
                    language="en",
                    prompt=whisper_prompt,
                )
                transcription = result["text"].strip().lower()
                transcription = apply_word_corrections(transcription)
                logger.info(f"Transcription: {transcription}")

                # if transcription.startswith(config.command):
                #     claude_command = transcription.removeprefix(config.command).removeprefix(",").strip()
                if transcription and transcription.lower() not in NOISE_PHRASES:
                    asyncio.create_task(_run_claude_task(transcription))

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        listener.stop()


async def run_whisper() -> None:
    """Main entry point for running Claude Whisper."""
    logger.info(f"Working directory: {Path.cwd()}")
    logger.info(f"Loading model: {config.model_name}")
    load_models.load_model(config.model_name)
    os.makedirs(config.plan_folder, exist_ok=True)

    await _run_audio_mode()


def main():
    parser = argparse.ArgumentParser(description="Claude Whisper - Voice-activated Claude AI assistant")
    parser.add_argument(
        "--working-directory",
        type=str,
        default=None,
        help="Working directory to use (defaults to current directory).",
    )
    parser.add_argument(
        "--push-to-talk-key",
        type=str,
        default=None,
        help="Key for push-to-talk (e.g., 'esc', 'space', 'ctrl'). Overrides config file and env var.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Claude model to use (e.g., 'claude-sonnet-4-5-20250929', 'claude-opus-4-6'). Overrides config file and env var.",
    )

    args = parser.parse_args()

    if args.working_directory:
        target = Path(args.working_directory).resolve()
        if not target.is_dir():
            parser.error(f"Working directory does not exist: {target}")
        config.cwd = target

    if args.push_to_talk_key:
        config.push_to_talk_key = args.push_to_talk_key

    if args.model:
        config.model = args.model

    asyncio.run(run_whisper())
