# Claude Whisper Test Suite

This directory contains the comprehensive test suite for the Claude Whisper project.

## Test Structure

```
tests/
├── __init__.py                # Test package initialization
├── conftest.py               # Shared fixtures and pytest configuration
├── test_config.py            # Tests for configuration management
├── test_main.py              # Tests for core application logic
└── test_integration.py       # Integration tests for main entry points
```

## Test Coverage

### Unit Tests

#### `test_config.py`
Tests for the `Config` class and configuration loading:
- Default values
- Environment variable loading (CLAUDE_WHISPER_ prefix)
- TOML file loading
- Configuration priority (init > env > toml > defaults)
- Push-to-talk key variations

#### `test_main.py`
Tests for core application components:
- **TaskType & TransientError Enums**: Value and member verification
- **TaskContext**: Dataclass initialization and field management
- **TaskTypeDetector**: Command pattern detection (plan vs edit)
- **_parse_push_to_talk_key**: Keyboard key parsing
- **LifecycleManager**: Context creation and lifecycle routing
- **BaseLifecycle**: Notification sending and error detection
- **EditLifecycle**: Command execution and transient error handling
- **PlanLifecycle**: Plan extraction and file saving
- **ClaudeSDKSession**: Session management and lifecycle orchestration
- **_run_claude_task**: Task execution with proper options

### Integration Tests

#### `test_integration.py`
Tests for main entry points and mode switching:
- **_run_audio_mode**: Audio recording and keyboard listener testing
- **run_whisper**: Model loading, path expansion, and mode routing

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/test_main.py
pytest tests/test_config.py
pytest tests/test_integration.py
```

### Run Specific Test Class
```bash
pytest tests/test_main.py::TestTaskTypeDetector
```

### Run Specific Test Method
```bash
pytest tests/test_main.py::TestTaskTypeDetector::test_detect_plan_patterns
```

### Run with Coverage Report
```bash
pytest --cov=claude_whisper --cov-report=html
```

The coverage report will be generated in `htmlcov/index.html`.

### Run Only Unit Tests
```bash
pytest -m unit
```

### Run Only Integration Tests
```bash
pytest -m integration
```

### Run Tests in Parallel
```bash
pytest -n auto
```

## Test Dependencies

The test suite requires the following dependencies (defined in `pyproject.toml`):
- `pytest>=8.0.0` - Testing framework
- `pytest-asyncio>=0.23.0` - Async test support
- `pytest-cov>=4.1.0` - Coverage reporting
- `pytest-mock>=3.12.0` - Enhanced mocking capabilities

Install test dependencies:
```bash
uv sync --dev
```

## Writing New Tests

### Test Naming Conventions
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Using Fixtures
Common fixtures are defined in `conftest.py`:
- `mock_config`: Mock configuration object
- `tmp_working_dir`: Temporary working directory
- `mock_claude_client`: Mock Claude SDK client
- `sample_task_context`: Sample TaskContext instance

Example:
```python
def test_something(mock_config, tmp_working_dir):
    # Use fixtures in your test
    assert mock_config.command == "jarvis"
    assert tmp_working_dir.exists()
```

### Async Tests
Use the `@pytest.mark.asyncio` decorator for async tests:
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None
```

### Mocking External Dependencies
Use `unittest.mock.patch` for mocking:
```python
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_with_mocks():
    with patch("module.external_call", new_callable=AsyncMock) as mock_call:
        await function_under_test()
        mock_call.assert_called_once()
```

## Test Organization

Tests are organized by the components they test:
1. **Enums and Constants**: Simple value checks
2. **Data Classes**: Initialization and field management
3. **Utility Functions**: Input/output behavior
4. **Business Logic Classes**: Core functionality
5. **Integration Points**: End-to-end workflows

## CI/CD Integration

These tests are designed to run in CI/CD pipelines. Key considerations:
- All tests are deterministic and repeatable
- External dependencies are mocked
- Tests clean up after themselves (using tmp_path fixtures)
- No network calls or file system dependencies outside of temp directories

## Coverage Goals

Target coverage metrics:
- Overall: >80%
- Core business logic: >90%
- Utility functions: 100%

Check current coverage:
```bash
pytest --cov=claude_whisper --cov-report=term-missing
```

## Troubleshooting

### Tests Fail with Import Errors
Ensure the package is installed in development mode:
```bash
uv sync --dev
```

### Async Tests Not Running
Make sure `pytest-asyncio` is installed and `asyncio_mode = auto` is set in `pytest.ini`.

### Mock Assertions Failing
Check that you're using the correct mock type:
- Use `AsyncMock` for async functions
- Use `MagicMock` or `Mock` for sync functions

### Audio Tests Failing
Audio-related tests mock PyAudio and audio processing. If tests fail, ensure:
- numpy is properly installed
- Mock audio data is in the correct format (int16)
