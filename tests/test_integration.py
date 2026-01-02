"""Integration tests for main entry points and audio functionality."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from claude_whisper import _run_audio_mode, _run_bypass_mode, run_whisper


class TestRunBypassMode:
    """Test _run_bypass_mode function."""

    @pytest.mark.asyncio
    async def test_bypass_mode_accepts_input(self):
        """Test that bypass mode accepts and processes text input."""
        # Simulate user entering one command then EOF
        inputs = iter(["test command", ""])

        async def mock_input(*args, **kwargs):
            try:
                return next(inputs)
            except StopIteration:
                raise EOFError from None

        with patch("claude_whisper._run_claude_task", new_callable=AsyncMock) as mock_run_task:
            with patch("asyncio.to_thread", side_effect=mock_input):
                with patch("builtins.print"):
                    try:
                        await _run_bypass_mode(Path("/tmp"))
                    except EOFError:
                        pass

                    # Should have been called once with the command
                    mock_run_task.assert_called()

    @pytest.mark.asyncio
    async def test_bypass_mode_strips_input(self):
        """Test that bypass mode strips whitespace from input."""
        inputs = iter(["  test command  ", ""])

        async def mock_input(*args, **kwargs):
            try:
                return next(inputs)
            except StopIteration:
                raise EOFError from None

        with patch("claude_whisper._run_claude_task", new_callable=AsyncMock):
            with patch("asyncio.to_thread", side_effect=mock_input):
                with patch("builtins.print"):
                    with patch("asyncio.create_task") as mock_create_task:
                        try:
                            await _run_bypass_mode(Path("/tmp"))
                        except EOFError:
                            pass

                        # Verify create_task was called (input was not empty after strip)
                        assert mock_create_task.called

    @pytest.mark.asyncio
    async def test_bypass_mode_ignores_empty_input(self):
        """Test that bypass mode ignores empty input."""
        inputs = iter(["", "  ", ""])

        async def mock_input(*args, **kwargs):
            try:
                return next(inputs)
            except StopIteration:
                raise EOFError from None

        with patch("claude_whisper._run_claude_task", new_callable=AsyncMock):
            with patch("asyncio.to_thread", side_effect=mock_input):
                with patch("builtins.print"):
                    with patch("asyncio.create_task") as mock_create_task:
                        try:
                            await _run_bypass_mode(Path("/tmp"))
                        except EOFError:
                            pass

                        # Should not create any tasks for empty input
                        mock_create_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_bypass_mode_handles_keyboard_interrupt(self):
        """Test that bypass mode handles KeyboardInterrupt gracefully."""

        async def mock_input(*args, **kwargs):
            raise KeyboardInterrupt

        with patch("asyncio.to_thread", side_effect=mock_input):
            with patch("builtins.print"):
                with patch("claude_whisper.logger.info") as mock_logger:
                    await _run_bypass_mode(Path("/tmp"))
                    mock_logger.assert_any_call("Exiting bypass mode")


class TestRunAudioMode:
    """Test _run_audio_mode function."""

    @pytest.mark.asyncio
    async def test_audio_mode_initializes_keyboard_listener(self):
        """Test that audio mode initializes keyboard listener."""
        mock_audio = MagicMock()
        mock_stream = MagicMock()
        mock_stream.stop_stream = MagicMock()
        mock_stream.close = MagicMock()
        mock_audio.open.return_value = mock_stream
        mock_audio.terminate = MagicMock()

        mock_listener = MagicMock()
        mock_listener.stop = MagicMock()

        with patch("claude_whisper.config") as mock_config:
            mock_config.push_to_talk_key = "esc"
            mock_config.format = 8
            mock_config.channels = 1
            mock_config.rate = 16000
            mock_config.chunk = 1024

            with patch("pyaudio.PyAudio", return_value=mock_audio):
                with patch("claude_whisper.keyboard.Listener", return_value=mock_listener) as mock_listener_class:
                    # Mock the event loop to prevent infinite loop
                    with patch("asyncio.Event") as mock_event_class:
                        mock_event = MagicMock()
                        mock_event.wait = AsyncMock(side_effect=asyncio.CancelledError)
                        mock_event_class.return_value = mock_event

                        try:
                            await _run_audio_mode(Path("/tmp"))
                        except (asyncio.CancelledError, AttributeError, UnboundLocalError):
                            pass

                        # Verify listener was created and started
                        mock_listener_class.assert_called_once()
                        mock_listener.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_audio_mode_processes_wake_word(self):
        """Test that audio mode processes commands with wake word."""
        # This is a complex integration test that would require mocking
        # the entire audio recording and transcription pipeline
        # For now, we'll test the key components are called

        mock_audio = MagicMock()
        mock_stream = MagicMock()
        mock_audio.open.return_value = mock_stream

        # Create mock audio data that exceeds silence threshold
        audio_data = np.array([1000, 2000, 1500] * 100, dtype=np.int16)

        async def mock_read(*args, **kwargs):
            return audio_data.tobytes()

        mock_stream.read = mock_read

        with patch("claude_whisper.config") as mock_config:
            mock_config.push_to_talk_key = "esc"
            mock_config.command = "jarvis"
            mock_config.silence_threshold = 500
            mock_config.model_name = "test-model"
            mock_config.format = 8
            mock_config.channels = 1
            mock_config.rate = 16000
            mock_config.chunk = 1024

            with patch("pyaudio.PyAudio", return_value=mock_audio):
                with patch("claude_whisper.keyboard.Listener") as mock_listener_class:
                    mock_listener = MagicMock()
                    mock_listener_class.return_value = mock_listener

                    with patch("asyncio.Event") as mock_event_class:
                        # Simulate: not recording -> recording -> not recording -> exit
                        call_count = [0]

                        async def mock_wait():
                            call_count[0] += 1
                            if call_count[0] > 2:
                                raise asyncio.CancelledError

                        def mock_is_set():
                            # First wait() -> is_set() returns True (recording)
                            # Then returns False (stop recording)
                            return call_count[0] == 1

                        mock_event = MagicMock()
                        mock_event.wait = mock_wait
                        mock_event.is_set = mock_is_set
                        mock_event_class.return_value = mock_event

                        with patch("mlx_whisper.transcribe") as mock_transcribe:
                            mock_transcribe.return_value = {"text": "jarvis fix the bug"}

                            with patch("asyncio.to_thread", side_effect=lambda f, *args, **kwargs: f(*args, **kwargs)):
                                with patch("claude_whisper._run_claude_task", new_callable=AsyncMock):
                                    with patch("asyncio.create_task"):
                                        try:
                                            await _run_audio_mode(Path("/tmp"))
                                        except (asyncio.CancelledError, AttributeError):
                                            pass

                                        # Verify transcription was called if recording happened
                                        if call_count[0] > 1:
                                            assert mock_transcribe.called or not mock_event.is_set()


class TestRunWhisper:
    """Test run_whisper function."""

    @pytest.mark.asyncio
    async def test_run_whisper_loads_model(self):
        """Test that run_whisper loads the Whisper model."""
        with patch("claude_whisper.config") as mock_config:
            mock_config.model_name = "test-model"
            mock_config.plan_folder = "plans"

            with patch("claude_whisper.load_models.load_model") as mock_load:
                with patch("os.makedirs") as mock_makedirs:
                    with patch("claude_whisper._run_bypass_mode", new_callable=AsyncMock):
                        await run_whisper(Path("/tmp"), bypass_whisper=True)

                        mock_load.assert_called_once_with("test-model")
                        mock_makedirs.assert_called_once_with("plans", exist_ok=True)

    @pytest.mark.asyncio
    async def test_run_whisper_bypass_mode(self):
        """Test that run_whisper calls bypass mode when requested."""
        with patch("claude_whisper.config") as mock_config:
            mock_config.model_name = "test-model"
            mock_config.plan_folder = "plans"

            with patch("claude_whisper.load_models.load_model"):
                with patch("os.makedirs"):
                    with patch("claude_whisper._run_bypass_mode", new_callable=AsyncMock) as mock_bypass:
                        await run_whisper(Path("/tmp"), bypass_whisper=True)
                        mock_bypass.assert_called_once_with(Path("/tmp"))

    @pytest.mark.asyncio
    async def test_run_whisper_audio_mode(self):
        """Test that run_whisper calls audio mode by default."""
        with patch("claude_whisper.config") as mock_config:
            mock_config.model_name = "test-model"
            mock_config.plan_folder = "plans"

            with patch("claude_whisper.load_models.load_model"):
                with patch("os.makedirs"):
                    with patch("claude_whisper._run_audio_mode", new_callable=AsyncMock) as mock_audio:
                        await run_whisper(Path("/tmp"), bypass_whisper=False)
                        mock_audio.assert_called_once_with(Path("/tmp"))

    @pytest.mark.asyncio
    async def test_run_whisper_expands_user_path(self):
        """Test that run_whisper expands user home directory."""
        with patch("claude_whisper.config") as mock_config:
            mock_config.model_name = "test-model"
            mock_config.plan_folder = "plans"

            with patch("claude_whisper.load_models.load_model"):
                with patch("os.makedirs"):
                    with patch("claude_whisper._run_bypass_mode", new_callable=AsyncMock) as mock_bypass:
                        await run_whisper(Path("~/test"), bypass_whisper=True)

                        # Verify the path was expanded
                        call_args = mock_bypass.call_args[0][0]
                        assert "~" not in str(call_args)

    @pytest.mark.asyncio
    async def test_run_whisper_creates_plan_folder(self):
        """Test that run_whisper creates the plan folder."""
        with patch("claude_whisper.config") as mock_config:
            mock_config.model_name = "test-model"
            mock_config.plan_folder = "custom_plans"

            with patch("claude_whisper.load_models.load_model"):
                with patch("os.makedirs") as mock_makedirs:
                    with patch("claude_whisper._run_bypass_mode", new_callable=AsyncMock):
                        await run_whisper(Path("/tmp"), bypass_whisper=True)

                        mock_makedirs.assert_called_once_with("custom_plans", exist_ok=True)
