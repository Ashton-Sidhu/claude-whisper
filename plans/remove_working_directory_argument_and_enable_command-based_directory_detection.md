# Plan: Remove Working Directory Argument and Enable Command-Based Directory Detection

## Overview
Refactor claude-whisper to remove the required `working_dir` positional argument and instead intelligently determine the working directory from the command itself, with a fallback to the current working directory where claude-whisper is launched.

## Current Architecture
The working directory flows through the system as follows:
1. **CLI Entry** (`__init__.py:381-390`): Parses `working_dir` as positional argument
2. **run_whisper()** (`__init__.py:367-378`): Receives and expands working_dir
3. **_run_audio_mode()** / **_run_bypass_mode()**: Pass working_dir to tasks
4. **_run_claude_task()** (`__init__.py:228-242`): Creates TaskContext and sets cwd in ClaudeAgentOptions
5. **TaskContext** (`__init__.py:38-50`): Stores working_dir for each task
6. **ClaudeSDKClient**: Uses cwd from ClaudeAgentOptions

## Design Approach

### Directory Detection Strategy
Implement a `DirectoryExtractor` class that:
1. Parses natural language commands for directory references
2. Supports common path patterns:
   - Absolute paths: `/path/to/project`
   - Home directory: `~/projects/myapp`
   - Relative paths: `./src` or `../otherproject`
   - Project references: "in my-project" or "in the backend directory"
3. Falls back to `Path.cwd()` if no directory is detected
4. Expands paths (e.g., `~` to home directory, relative to absolute)

### Implementation Changes

#### 1. Create Directory Extractor (`__init__.py`)
Add a new class `DirectoryExtractor` with method:
```python
def extract_directory(self, command: str) -> tuple[str, Path]:
    """
    Extract directory from command and return (cleaned_command, working_dir).
    Returns command with directory references removed and the detected path.
    """
```

Pattern matching for paths:
- Regex patterns to detect paths in natural language
- Keywords: "in", "at", "from", "inside", "within" + path
- Direct path patterns: `/...`, `~/...`, `./...`
- Fallback: `Path.cwd()`

#### 2. Update CLI Entry Point (`__init__.py:381-390`)
**Before:**
```python
def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument("working_dir", type=Path, help="Working directory...")
    parser.add_argument("--bypass-whisper", ...)
    args = parser.parse_args()
    asyncio.run(run_whisper(args.working_dir, args.bypass_whisper))
```

**After:**
```python
def main():
    parser = argparse.ArgumentParser(...)
    # Remove working_dir positional argument
    parser.add_argument("--bypass-whisper", ...)
    args = parser.parse_args()
    asyncio.run(run_whisper(args.bypass_whisper))
```

#### 3. Update run_whisper() (`__init__.py:367-378`)
**Before:**
```python
async def run_whisper(working_dir: Path, bypass_whisper: bool = False):
    working_dir = Path(working_dir).expanduser()
    ...
    if bypass_whisper:
        await _run_bypass_mode(working_dir)
    else:
        await _run_audio_mode(working_dir)
```

**After:**
```python
async def run_whisper(bypass_whisper: bool = False):
    # Remove working_dir parameter entirely
    # Modes will extract directory per command
    ...
    if bypass_whisper:
        await _run_bypass_mode()
    else:
        await _run_audio_mode()
```

#### 4. Update _run_bypass_mode() (`__init__.py:245-259`)
**Before:**
```python
async def _run_bypass_mode(working_dir: Path):
    ...
    asyncio.create_task(_run_claude_task(user_input, working_dir))
```

**After:**
```python
async def _run_bypass_mode():
    extractor = DirectoryExtractor()
    ...
    # Extract directory from each command
    cleaned_command, working_dir = extractor.extract_directory(user_input)
    asyncio.create_task(_run_claude_task(cleaned_command, working_dir))
```

#### 5. Update _run_audio_mode() (`__init__.py:287-364`)
**Before:**
```python
async def _run_audio_mode(working_dir: Path):
    ...
    asyncio.create_task(_run_claude_task(transcription, working_dir))
```

**After:**
```python
async def _run_audio_mode():
    extractor = DirectoryExtractor()
    ...
    # Extract directory from transcribed command
    cleaned_command, working_dir = extractor.extract_directory(transcription)
    asyncio.create_task(_run_claude_task(cleaned_command, working_dir))
```

#### 6. Keep _run_claude_task() Unchanged
This function already accepts `working_dir: Path` per command, so no changes needed. It will continue to receive the extracted directory from the modes above.

#### 7. Update README.md
Update usage documentation from:
```bash
claude-whisper /path/to/your/project
```

To:
```bash
# Launch from anywhere
claude-whisper

# Example commands:
# "fix the bug in ~/projects/myapp"  -> works in ~/projects/myapp
# "refactor authentication"          -> works in current directory
# "add tests to /Users/me/backend"   -> works in /Users/me/backend
```

## Critical Files to Modify
1. `/Users/sidhu/claude-whisper/src/claude_whisper/__init__.py` - Main implementation
2. `/Users/sidhu/claude-whisper/README.md` - Documentation update

## Testing Considerations
- Test directory extraction with various command patterns
- Test fallback to current working directory
- Test path expansion (~ and relative paths)
- Update existing tests that pass `working_dir` parameter
- Test files: `/Users/sidhu/claude-whisper/tests/test_main.py`, `/Users/sidhu/claude-whisper/tests/test_integration.py`

## Edge Cases to Handle
1. Multiple paths in one command: Use the first detected path
2. Invalid paths: Validate path exists, fallback to cwd if invalid
3. Ambiguous references: "in my project" without clear path -> use cwd
4. Empty commands: Use cwd
5. Path-like text that isn't a path: Be conservative in detection

## Benefits
- More natural voice commands: "fix the bug in my backend project"
- No need to specify directory at launch
- Different commands can work in different directories
- More flexible workflow for multi-project work
