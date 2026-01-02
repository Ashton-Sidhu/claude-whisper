"""Shared pytest fixtures and configuration for the test suite."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_config():
    """Provide a mock config object with default values."""
    config = MagicMock()
    config.model_name = "mlx-community/whisper-medium-mlx-8bit"
    config.format = 8
    config.channels = 1
    config.rate = 16000
    config.chunk = 1024
    config.silence_threshold = 500
    config.silence_chunks = 30
    config.command = "jarvis"
    config.tmux_session_name = "claude"
    config.permission_mode = "acceptEdits"
    config.push_to_talk_key = "esc"
    config.plan_folder = "plans"
    return config


@pytest.fixture
def tmp_working_dir(tmp_path):
    """Provide a temporary working directory for tests."""
    working_dir = tmp_path / "workspace"
    working_dir.mkdir()
    return working_dir


@pytest.fixture
def mock_claude_client():
    """Provide a mock Claude SDK client."""
    from unittest.mock import AsyncMock

    client = AsyncMock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.query = AsyncMock()
    client.interrupt = AsyncMock()

    async def mock_receive():
        return
        yield  # Make it an async generator

    client.receive_response = mock_receive
    return client


@pytest.fixture
def sample_task_context():
    """Provide a sample TaskContext for testing."""
    from claude_whisper import TaskContext, TaskType

    return TaskContext(
        task_id="test-id-123",
        task_type=TaskType.EDIT,
        command="test command",
        working_dir=Path("/tmp/test"),
        permission_mode="acceptEdits",
    )


# Configure pytest-asyncio
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "asyncio: mark test as an async test")
