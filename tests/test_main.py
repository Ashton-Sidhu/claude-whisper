"""Test suite for main application logic."""

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, ToolUseBlock
from pynput import keyboard

from claude_whisper import (
    BaseLifecycle,
    ClaudeSDKSession,
    EditLifecycle,
    LifecycleManager,
    PlanLifecycle,
    TaskContext,
    TaskType,
    TaskTypeDetector,
    TransientError,
    _parse_push_to_talk_key,
    _run_claude_task,
)


class TestTaskType:
    """Test TaskType enum."""

    def test_task_type_values(self):
        """Test that TaskType enum has correct values."""
        assert TaskType.PLAN.value == "plan"
        assert TaskType.EDIT.value == "edit"

    def test_task_type_members(self):
        """Test that all expected TaskType members exist."""
        assert hasattr(TaskType, "PLAN")
        assert hasattr(TaskType, "EDIT")


class TestTransientError:
    """Test TransientError enum."""

    def test_transient_error_values(self):
        """Test that TransientError enum has correct values."""
        assert TransientError.API == "API Error"

    def test_transient_error_is_string(self):
        """Test that TransientError behaves as a string."""
        assert isinstance(TransientError.API, str)
        assert TransientError.API.startswith("API")


class TestTaskContext:
    """Test TaskContext dataclass."""

    def test_task_context_creation(self):
        """Test creating a TaskContext with required fields."""
        ctx = TaskContext(
            task_id="test-id",
            task_type=TaskType.EDIT,
            command="test command",
            working_dir=Path("/tmp"),
            permission_mode="acceptEdits",
        )

        assert ctx.task_id == "test-id"
        assert ctx.task_type == TaskType.EDIT
        assert ctx.command == "test command"
        assert ctx.working_dir == Path("/tmp")
        assert ctx.permission_mode == "acceptEdits"
        assert isinstance(ctx.started_at, datetime)
        assert ctx.output is None
        assert ctx.finished_at is None
        assert ctx.error is None

    def test_task_context_with_optional_fields(self):
        """Test TaskContext with optional fields set."""
        now = datetime.now()
        error = Exception("test error")

        ctx = TaskContext(
            task_id="test-id",
            task_type=TaskType.PLAN,
            command="test command",
            working_dir=Path("/tmp"),
            permission_mode="plan",
            output="test output",
            finished_at=now,
            error=error,
        )

        assert ctx.output == "test output"
        assert ctx.finished_at == now
        assert ctx.error == error

    def test_task_context_started_at_default(self):
        """Test that started_at is automatically set to current time."""
        before = datetime.now()
        ctx = TaskContext(
            task_id="test-id",
            task_type=TaskType.EDIT,
            command="test",
            working_dir=Path("/tmp"),
            permission_mode="acceptEdits",
        )
        after = datetime.now()

        assert before <= ctx.started_at <= after


class TestTaskTypeDetector:
    """Test TaskTypeDetector class."""

    def test_detect_with_explicit_plan_permission(self):
        """Test that explicit plan permission mode takes priority."""
        detector = TaskTypeDetector()
        result = detector.detect("fix the bug", permission_mode="plan")
        assert result == TaskType.PLAN

    def test_detect_plan_patterns(self):
        """Test detection of plan-related patterns."""
        detector = TaskTypeDetector()

        plan_commands = [
            "can you plan the implementation",
            "design a new feature",
            "architect the system",
            "propose a solution",
            "what's the strategy for this",
            "suggest an approach for solving this",
            "create a blueprint for the feature",
        ]

        for command in plan_commands:
            result = detector.detect(command)
            assert result == TaskType.PLAN, f"Failed to detect plan pattern in: {command}"

    def test_detect_edit_patterns(self):
        """Test detection of edit-related patterns."""
        detector = TaskTypeDetector()

        edit_commands = [
            "fix the authentication bug",
            "update the config file",
            "modify the function",
            "change the variable name",
            "refactor this code",
            "add a new method to the class",
            "remove the unused imports",
            "delete the deprecated function",
            "edit the README",
            "implement the new feature",
            "create a new file",
            "write some tests",
            "replace the old logic",
            "rename the variable",
        ]

        for command in edit_commands:
            result = detector.detect(command)
            assert result == TaskType.EDIT, f"Failed to detect edit pattern in: {command}"

    def test_detect_case_insensitive(self):
        """Test that pattern matching is case-insensitive."""
        detector = TaskTypeDetector()

        assert detector.detect("PLAN the feature") == TaskType.PLAN
        assert detector.detect("Plan The Feature") == TaskType.PLAN
        assert detector.detect("FIX the bug") == TaskType.EDIT
        assert detector.detect("Fix The Bug") == TaskType.EDIT

    def test_detect_default_to_edit(self):
        """Test that unknown commands default to EDIT."""
        detector = TaskTypeDetector()

        ambiguous_commands = [
            "hello",
            "what is this",
            "explain the code",
            "show me the logs",
        ]

        for command in ambiguous_commands:
            result = detector.detect(command)
            assert result == TaskType.EDIT, f"Should default to EDIT for: {command}"

    def test_detect_empty_string(self):
        """Test handling of empty string."""
        detector = TaskTypeDetector()
        result = detector.detect("")
        assert result == TaskType.EDIT

    def test_detect_permission_mode_overrides_pattern(self):
        """Test that permission_mode overrides pattern matching."""
        detector = TaskTypeDetector()

        # Even with edit pattern, plan permission should win
        result = detector.detect("fix the bug", permission_mode="plan")
        assert result == TaskType.PLAN

        # Non-plan permission should fall back to pattern matching
        result = detector.detect("plan the feature", permission_mode="acceptEdits")
        assert result == TaskType.PLAN


class TestParsePushToTalkKey:
    """Test _parse_push_to_talk_key function."""

    def test_parse_esc_key(self):
        """Test parsing escape key."""
        assert _parse_push_to_talk_key("esc") == keyboard.Key.esc
        assert _parse_push_to_talk_key("escape") == keyboard.Key.esc
        assert _parse_push_to_talk_key("ESC") == keyboard.Key.esc
        assert _parse_push_to_talk_key("Escape") == keyboard.Key.esc

    def test_parse_space_key(self):
        """Test parsing space key."""
        assert _parse_push_to_talk_key("space") == keyboard.Key.space
        assert _parse_push_to_talk_key("SPACE") == keyboard.Key.space

    def test_parse_enter_key(self):
        """Test parsing enter key."""
        assert _parse_push_to_talk_key("enter") == keyboard.Key.enter
        assert _parse_push_to_talk_key("ENTER") == keyboard.Key.enter

    def test_parse_tab_key(self):
        """Test parsing tab key."""
        assert _parse_push_to_talk_key("tab") == keyboard.Key.tab
        assert _parse_push_to_talk_key("TAB") == keyboard.Key.tab

    def test_parse_modifier_keys(self):
        """Test parsing modifier keys."""
        assert _parse_push_to_talk_key("ctrl") == keyboard.Key.ctrl_l
        assert _parse_push_to_talk_key("shift") == keyboard.Key.shift_l
        assert _parse_push_to_talk_key("alt") == keyboard.Key.alt_l
        assert _parse_push_to_talk_key("cmd") == keyboard.Key.cmd_l

    def test_parse_single_character(self):
        """Test parsing single character keys."""
        result = _parse_push_to_talk_key("a")
        assert isinstance(result, keyboard.KeyCode)
        assert result.char == "a"

        result = _parse_push_to_talk_key("z")
        assert isinstance(result, keyboard.KeyCode)
        assert result.char == "z"

    def test_parse_unknown_key_defaults_to_esc(self):
        """Test that unknown keys default to ESC."""
        assert _parse_push_to_talk_key("unknown") == keyboard.Key.esc
        assert _parse_push_to_talk_key("f12") == keyboard.Key.esc
        assert _parse_push_to_talk_key("ctrl+shift") == keyboard.Key.esc

    def test_parse_with_whitespace(self):
        """Test parsing with leading/trailing whitespace."""
        assert _parse_push_to_talk_key("  space  ") == keyboard.Key.space
        assert _parse_push_to_talk_key("\tenter\t") == keyboard.Key.enter


class TestLifecycleManager:
    """Test LifecycleManager class."""

    def test_create_context_with_plan_permission(self):
        """Test creating context with plan permission mode."""
        manager = LifecycleManager()
        ctx = manager.create_context(
            command="fix the bug",
            working_dir=Path("/tmp"),
            permission_mode="plan",
        )

        assert ctx.task_type == TaskType.PLAN
        assert ctx.command == "fix the bug"
        assert ctx.working_dir == Path("/tmp")
        assert ctx.permission_mode == "plan"
        # task_id should be a valid UUID
        UUID(ctx.task_id)

    def test_create_context_with_plan_pattern(self):
        """Test creating context with plan pattern in command."""
        manager = LifecycleManager()
        ctx = manager.create_context(
            command="design a new feature",
            working_dir=Path("/tmp"),
            permission_mode="acceptEdits",
        )

        assert ctx.task_type == TaskType.PLAN

    def test_create_context_with_edit_pattern(self):
        """Test creating context with edit pattern in command."""
        manager = LifecycleManager()
        ctx = manager.create_context(
            command="fix the bug",
            working_dir=Path("/tmp"),
            permission_mode="acceptEdits",
        )

        assert ctx.task_type == TaskType.EDIT

    def test_get_lifecycle_for_plan(self):
        """Test getting lifecycle handler for plan task."""
        manager = LifecycleManager()
        ctx = TaskContext(
            task_id="test",
            task_type=TaskType.PLAN,
            command="test",
            working_dir=Path("/tmp"),
            permission_mode="plan",
        )

        lifecycle = manager.get_lifecycle(ctx)
        assert isinstance(lifecycle, PlanLifecycle)

    def test_get_lifecycle_for_edit(self):
        """Test getting lifecycle handler for edit task."""
        manager = LifecycleManager()
        ctx = TaskContext(
            task_id="test",
            task_type=TaskType.EDIT,
            command="test",
            working_dir=Path("/tmp"),
            permission_mode="acceptEdits",
        )

        lifecycle = manager.get_lifecycle(ctx)
        assert isinstance(lifecycle, EditLifecycle)

    def test_create_context_generates_unique_ids(self):
        """Test that each context gets a unique task_id."""
        manager = LifecycleManager()

        ctx1 = manager.create_context("test", Path("/tmp"), "acceptEdits")
        ctx2 = manager.create_context("test", Path("/tmp"), "acceptEdits")

        assert ctx1.task_id != ctx2.task_id


class TestBaseLifecycle:
    """Test BaseLifecycle class."""

    @pytest.mark.asyncio
    async def test_on_start_sends_notification(self):
        """Test that on_start sends a notification."""
        lifecycle = BaseLifecycle()
        ctx = TaskContext(
            task_id="test",
            task_type=TaskType.EDIT,
            command="test command that is very long and will be truncated",
            working_dir=Path("/tmp"),
            permission_mode="acceptEdits",
        )

        with patch("claude_whisper.notifier.send", new_callable=AsyncMock) as mock_send:
            await lifecycle.on_start(ctx)
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args.kwargs["title"] == "Task started"
            assert "test command that is very long and will be truncat..." in call_args.kwargs["message"]

    @pytest.mark.asyncio
    async def test_on_finish_sends_notification(self):
        """Test that on_finish sends a notification."""
        lifecycle = BaseLifecycle()
        ctx = TaskContext(
            task_id="test",
            task_type=TaskType.EDIT,
            command="test command",
            working_dir=Path("/tmp"),
            permission_mode="acceptEdits",
        )

        with patch("claude_whisper.notifier.send", new_callable=AsyncMock) as mock_send:
            await lifecycle.on_finish(ctx)
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args.kwargs["title"] == "Task Finished"

    @pytest.mark.asyncio
    async def test_on_error_sends_notification_and_logs(self):
        """Test that on_error sends notification and logs error."""
        lifecycle = BaseLifecycle()
        ctx = TaskContext(
            task_id="test-123",
            task_type=TaskType.EDIT,
            command="test",
            working_dir=Path("/tmp"),
            permission_mode="acceptEdits",
        )

        with patch("claude_whisper.notifier.send", new_callable=AsyncMock) as mock_send:
            with patch("claude_whisper.logger.error") as mock_logger:
                await lifecycle.on_error(ctx, "Test error message")
                mock_send.assert_called_once()
                mock_logger.assert_called_once_with("Test error message")

    def test_is_transient_error_with_api_error(self):
        """Test detecting transient API errors."""
        lifecycle = BaseLifecycle()
        block = TextBlock(text="API Error: Something went wrong")

        assert lifecycle.is_transient_error(block) is True

    def test_is_transient_error_with_normal_text(self):
        """Test that normal text is not detected as transient error."""
        lifecycle = BaseLifecycle()
        block = TextBlock(text="This is normal output")

        assert lifecycle.is_transient_error(block) is False


class TestEditLifecycle:
    """Test EditLifecycle class."""

    @pytest.mark.asyncio
    async def test_execute_submits_command(self):
        """Test that execute submits the command to the client."""
        lifecycle = EditLifecycle()
        ctx = TaskContext(
            task_id="test",
            task_type=TaskType.EDIT,
            command="fix the bug",
            working_dir=Path("/tmp"),
            permission_mode="acceptEdits",
        )

        # Create async generator for empty response
        async def mock_response():
            return
            yield  # Make it an async generator

        mock_client = AsyncMock()
        mock_client.query = AsyncMock()
        mock_client.receive_response = mock_response

        await lifecycle.execute(ctx, mock_client)

        mock_client.query.assert_called_once_with("fix the bug")

    @pytest.mark.asyncio
    async def test_execute_handles_text_blocks(self):
        """Test that execute handles text blocks in response."""
        lifecycle = EditLifecycle()
        ctx = TaskContext(
            task_id="test",
            task_type=TaskType.EDIT,
            command="test",
            working_dir=Path("/tmp"),
            permission_mode="acceptEdits",
        )

        # Create async generator for mock response
        async def mock_response():
            text_block = TextBlock(text="Some output")
            message = AssistantMessage(content=[text_block], model="claude-sonnet-4")
            yield message

        mock_client = AsyncMock()
        mock_client.query = AsyncMock()
        mock_client.receive_response = mock_response

        with patch("claude_whisper.logger.debug") as mock_logger:
            await lifecycle.execute(ctx, mock_client)
            mock_logger.assert_called()

    @pytest.mark.asyncio
    async def test_execute_handles_transient_errors(self):
        """Test that execute handles transient API errors."""
        lifecycle = EditLifecycle()
        ctx = TaskContext(
            task_id="test",
            task_type=TaskType.EDIT,
            command="test",
            working_dir=Path("/tmp"),
            permission_mode="acceptEdits",
        )

        # Create async generator with API error
        async def mock_response():
            error_block = TextBlock(text="API Error: Rate limit exceeded")
            message = AssistantMessage(content=[error_block], model="claude-sonnet-4")
            yield message

        mock_client = AsyncMock()
        mock_client.query = AsyncMock()
        mock_client.receive_response = mock_response
        mock_client.interrupt = AsyncMock()

        with patch.object(lifecycle, "on_error", new_callable=AsyncMock) as mock_on_error:
            await lifecycle.execute(ctx, mock_client)
            mock_on_error.assert_called_once()
            mock_client.interrupt.assert_called_once()


class TestPlanLifecycle:
    """Test PlanLifecycle class."""

    @pytest.mark.asyncio
    async def test_execute_extracts_plan_from_tool_use(self):
        """Test that execute extracts plan from ExitPlanMode tool use."""
        lifecycle = PlanLifecycle()
        ctx = TaskContext(
            task_id="test",
            task_type=TaskType.PLAN,
            command="design a feature",
            working_dir=Path("/tmp"),
            permission_mode="plan",
        )

        # Create async generator with tool use block
        async def mock_response():
            tool_block = ToolUseBlock(
                id="tool-1",
                name="ExitPlanMode",
                input={"plan": "# Plan: Test Feature\n\nThis is a test plan."},
            )
            message = AssistantMessage(content=[tool_block], model="claude-sonnet-4")
            yield message

        mock_client = AsyncMock()
        mock_client.query = AsyncMock()
        mock_client.receive_response = mock_response

        await lifecycle.execute(ctx, mock_client)

        assert ctx.output == "# Plan: Test Feature\n\nThis is a test plan."

    @pytest.mark.asyncio
    async def test_execute_handles_missing_plan(self):
        """Test that execute handles missing plan in tool use."""
        lifecycle = PlanLifecycle()
        ctx = TaskContext(
            task_id="test",
            task_type=TaskType.PLAN,
            command="design a feature",
            working_dir=Path("/tmp"),
            permission_mode="plan",
        )

        # Create async generator with tool use but no plan
        async def mock_response():
            tool_block = ToolUseBlock(
                id="tool-1",
                name="ExitPlanMode",
                input={},  # No plan
            )
            message = AssistantMessage(content=[tool_block], model="claude-sonnet-4")
            yield message

        mock_client = AsyncMock()
        mock_client.query = AsyncMock()
        mock_client.receive_response = mock_response

        await lifecycle.execute(ctx, mock_client)

        assert ctx.output == ""

    @pytest.mark.asyncio
    async def test_on_finish_saves_plan_to_file(self, tmp_path):
        """Test that on_finish saves the plan to a file."""
        lifecycle = PlanLifecycle()
        ctx = TaskContext(
            task_id="test",
            task_type=TaskType.PLAN,
            command="design a feature",
            working_dir=Path("/tmp"),
            permission_mode="plan",
            output="# Plan: Test Feature\n\nThis is a test plan.",
        )

        # Create temporary plans directory
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()

        with patch("claude_whisper.notifier.send", new_callable=AsyncMock):
            with patch("builtins.open", create=True) as mock_open:
                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file

                await lifecycle.on_finish(ctx)

                mock_open.assert_called_once_with("plans/test_feature.md", "w")
                mock_file.write.assert_called_once_with("# Plan: Test Feature\n\nThis is a test plan.")

    @pytest.mark.asyncio
    async def test_on_finish_handles_no_output(self):
        """Test that on_finish handles case with no plan output."""
        lifecycle = PlanLifecycle()
        ctx = TaskContext(
            task_id="test",
            task_type=TaskType.PLAN,
            command="design a feature",
            working_dir=Path("/tmp"),
            permission_mode="plan",
            output="",
        )

        with patch("claude_whisper.logger.info") as mock_logger:
            await lifecycle.on_finish(ctx)
            # When output is empty string, it's falsy so logger is called
            mock_logger.assert_any_call("No plan to save")


class TestClaudeSDKSession:
    """Test ClaudeSDKSession class."""

    @pytest.mark.asyncio
    async def test_session_initialization(self):
        """Test that session initializes correctly."""
        options = ClaudeAgentOptions(
            allowed_tools=["Read", "Write"],
            permission_mode="acceptEdits",
            cwd=Path("/tmp"),
        )
        ctx = TaskContext(
            task_id="test",
            task_type=TaskType.EDIT,
            command="test",
            working_dir=Path("/tmp"),
            permission_mode="acceptEdits",
        )

        with patch("claude_whisper.ClaudeSDKClient") as mock_client_class:
            session = ClaudeSDKSession(options, ctx)

            assert session.ctx == ctx
            assert isinstance(session.lifecycle, EditLifecycle)
            mock_client_class.assert_called_once_with(options)

    @pytest.mark.asyncio
    async def test_run_executes_full_lifecycle(self):
        """Test that run executes the full lifecycle."""
        options = ClaudeAgentOptions(
            allowed_tools=["Read", "Write"],
            permission_mode="acceptEdits",
            cwd=Path("/tmp"),
        )
        ctx = TaskContext(
            task_id="test",
            task_type=TaskType.EDIT,
            command="test",
            working_dir=Path("/tmp"),
            permission_mode="acceptEdits",
        )

        with patch("claude_whisper.ClaudeSDKClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            session = ClaudeSDKSession(options, ctx)

            with patch.object(session.lifecycle, "on_start", new_callable=AsyncMock) as mock_on_start:
                with patch.object(session.lifecycle, "execute", new_callable=AsyncMock) as mock_execute:
                    with patch.object(session.lifecycle, "on_finish", new_callable=AsyncMock) as mock_on_finish:
                        await session.run("test command")

                        mock_on_start.assert_called_once_with(ctx)
                        mock_client.connect.assert_called_once()
                        mock_execute.assert_called_once_with(ctx, mock_client)
                        mock_client.disconnect.assert_called_once()
                        mock_on_finish.assert_called_once_with(ctx)


class TestRunClaudeTask:
    """Test _run_claude_task function."""

    @pytest.mark.asyncio
    async def test_run_claude_task_creates_options(self):
        """Test that _run_claude_task creates ClaudeAgentOptions correctly."""
        with patch("claude_whisper.config") as mock_config:
            mock_config.permission_mode = "acceptEdits"
            mock_config.plan_folder = "plans"

            with patch("claude_whisper.ClaudeSDKSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_session.run = AsyncMock()
                mock_session_class.return_value = mock_session

                await _run_claude_task("test command", Path("/tmp"))

                # Verify session was created and run was called
                mock_session_class.assert_called_once()
                mock_session.run.assert_called_once_with("test command")

                # Check the options passed to session
                call_args = mock_session_class.call_args[0]
                options = call_args[0]
                assert options.allowed_tools == ["Read", "Write", "Bash"]
                assert options.permission_mode == "acceptEdits"
                assert options.cwd == Path("/tmp")
