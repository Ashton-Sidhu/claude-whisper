# Plan: Natural Language File Reference Mapping System

## Overview

Implement a system that detects natural language file references in user commands (e.g., "the main file", "config file") and automatically maps them to actual file paths before passing commands to Claude. This enables users to speak naturally about files without needing to know exact paths.

## Architecture

### Core Components

1. **FileResolver** - Maps reference types to actual file paths using priority-based matching
2. **FileReferenceDetector** - Detects natural language patterns in user commands
3. **FileReferenceReplacer** - Replaces detected references with actual paths
4. **Enhanced BaseLifecycle** - Integrates the system into the pre_execute() hook

### Integration Point

The system hooks into `BaseLifecycle.pre_execute()` to process commands before Claude sees them. This preserves the existing architecture while adding new functionality transparently.

## Implementation Details

### 1. FileResolver Class

**Location**: `src/claude_whisper/file_references.py` (new file)

**Responsibilities**:
- Scan working directory for files (with caching)
- Resolve specific file types (main, config, test, init)
- Generic keyword-based resolution
- Priority-based ranking of matches

**Key Methods**:

```python
class FileResolver:
    def __init__(self, working_dir: Path):
        self.working_dir = working_dir
        self._file_cache: Optional[list[Path]] = None

    def _scan_directory(self, max_depth: int = 3) -> list[Path]:
        """Scan directory, cache results, ignore .git/.venv/etc."""
        # Returns all files up to max_depth, cached per command

    def resolve_main_file(self) -> list[Path]:
        """Priority: __main__.py > main.py > app.py > __init__.py"""

    def resolve_config_file(self) -> list[Path]:
        """Priority: config.py > config.toml > settings.py > .env"""

    def resolve_test_file(self, target_name: Optional[str] = None) -> list[Path]:
        """Find test_*.py or *_test.py, optionally filtered by target"""

    def resolve_init_file(self) -> list[Path]:
        """Priority: __init__.py > init.py > setup.py"""

    def resolve_by_keyword(self, keyword: str) -> list[Path]:
        """Generic: models.py > models_*.py > *models*.py"""
```

**Priority System**: Each resolver returns files sorted by priority (10=exact match, 5=partial match)

### 2. FileReferenceDetector Class

**Location**: `src/claude_whisper/file_references.py`

**Responsibilities**:
- Define regex patterns for each file reference type
- Extract all matches from command string
- Return DetectedReference objects with position info
- Remove overlapping matches

**Supported Patterns**:

| Pattern | Regex | Maps To |
|---------|-------|---------|
| "the main file", "main file" | `\bthe\s+main\s+file\b` | resolve_main_file() |
| "config file", "settings file" | `\bconfig(?:uration)?\s+file\b` | resolve_config_file() |
| "test file", "test file for X" | `\btest\s+file(?:\s+for\s+(\w+))?\b` | resolve_test_file(X) |
| "init file", "__init__ file" | `\binit(?:ialization)?\s+file\b` | resolve_init_file() |
| "the X file" (generic) | `\bthe\s+(\w+)\s+file\b` | resolve_by_keyword(X) |

**DetectedReference Dataclass**:

```python
@dataclass
class DetectedReference:
    original_text: str        # "the config file"
    start_pos: int            # Position in command
    end_pos: int
    resolved_paths: list[Path]  # Matching files
    confidence: str           # "high", "medium", "low"
    reference_type: str       # "main", "config", "test", etc.
```

**Key Methods**:

```python
class FileReferenceDetector:
    def __init__(self, resolver: FileResolver):
        self.resolver = resolver
        self._compile_patterns()  # Compile all regex patterns

    def detect(self, command: str) -> list[DetectedReference]:
        """Detect all file references, remove overlaps"""
        # Returns prioritized, non-overlapping matches
```

### 3. FileReferenceReplacer Class

**Location**: `src/claude_whisper/file_references.py`

**Responsibilities**:
- Replace detected references with actual paths
- Handle ambiguity (multiple matches) based on strategy
- Generate metadata about replacements

**Ambiguity Strategies**:

```python
class AmbiguityResolutionStrategy(Enum):
    USE_FIRST = "use_first"  # Use highest priority match (default)
    LIST_ALL = "list_all"    # Replace with "(path1 or path2)"
    SKIP = "skip"            # Leave ambiguous references unchanged
```

**Key Methods**:

```python
class FileReferenceReplacer:
    def __init__(self, working_dir: Path,
                 strategy: AmbiguityResolutionStrategy = USE_FIRST):
        self.working_dir = working_dir
        self.strategy = strategy

    def replace_references(self, command: str,
                          detected_refs: list[DetectedReference]) -> tuple[str, dict]:
        """
        Returns: (modified_command, metadata)

        Metadata structure:
        {
            'replacements_made': 2,
            'replacements': [
                {
                    'original': 'the config file',
                    'replacement': 'src/app/config.py',
                    'paths': ['src/app/config.py'],
                    'confidence': 'high',
                    'type': 'config'
                }
            ]
        }
        """
```

### 4. Enhanced TaskContext

**Location**: `src/claude_whisper/__init__.py` (modify existing)

Add new field to TaskContext dataclass:

```python
@dataclass
class TaskContext:
    # ... existing fields ...

    # NEW: Track file reference replacements
    file_references_metadata: dict = field(default_factory=dict)
```

### 5. Enhanced BaseLifecycle

**Location**: `src/claude_whisper/__init__.py` (modify existing)

Update `pre_execute()` method:

```python
class BaseLifecycle:
    async def pre_execute(self, ctx: TaskContext):
        """Pre-execution hook with file reference resolution."""
        # Create resolver and detector
        resolver = FileResolver(ctx.working_dir)
        detector = FileReferenceDetector(resolver)
        replacer = FileReferenceReplacer(
            ctx.working_dir,
            strategy=AmbiguityResolutionStrategy.USE_FIRST
        )

        # Detect and replace
        detected_refs = detector.detect(ctx.command)

        if detected_refs:
            logger.info(f"Detected {len(detected_refs)} file reference(s)")
            modified_command, metadata = replacer.replace_references(
                ctx.command, detected_refs
            )

            # Update context
            ctx.command = modified_command
            ctx.file_references_metadata = metadata

            # Log replacements
            for repl in metadata.get('replacements', []):
                logger.info(
                    f"Replaced '{repl['original']}' with '{repl['replacement']}'"
                )
```

**Note**: PlanLifecycle and EditLifecycle inherit from BaseLifecycle, so they automatically get this functionality.

### 6. Configuration Options

**Location**: `src/claude_whisper/config.py` (modify existing)

Add new configuration fields:

```python
class Config(BaseSettings):
    # ... existing fields ...

    # File reference mapping settings
    file_reference_enabled: bool = Field(
        default=True,
        description="Enable natural language file reference mapping"
    )

    file_reference_strategy: str = Field(
        default="use_first",
        description="Ambiguity handling: use_first, list_all, skip"
    )

    file_reference_max_depth: int = Field(
        default=3,
        description="Maximum directory depth for file scanning"
    )
```

## Behavior Examples

### Example 1: Single Match (High Confidence)

```
Input:  "update the config file to add new settings"
Scan:   Found 1 file: src/claude_whisper/config.py
Output: "update src/claude_whisper/config.py to add new settings"
Log:    INFO: Replaced 'the config file' with 'src/claude_whisper/config.py'
```

### Example 2: Multiple Matches (Medium Confidence)

```
Input:  "read the main file"
Scan:   Found 2 files: src/claude_whisper/__main__.py (priority 10),
                       src/claude_whisper/__init__.py (priority 7)
Output: "read src/claude_whisper/__main__.py" (uses highest priority)
Log:    WARNING: Multiple matches for 'the main file', using: __main__.py
```

### Example 3: No Matches (Low Confidence)

```
Input:  "update the database file"
Scan:   No files matching "database"
Output: "update the database file" (unchanged)
Log:    WARNING: No files found for reference: the database file
```

### Example 4: Multiple References

```
Input:  "copy the config file settings to the main file"
Scan:   Found config.py, __main__.py
Output: "copy the config.py settings to the __main__.py"
Log:    INFO: Replaced 2 file reference(s)
```

## File Structure Changes

### New Files

- `src/claude_whisper/file_references.py` - All file reference classes (~400 lines)
- `tests/test_file_references.py` - Comprehensive test suite (~800 lines)

### Modified Files

- `src/claude_whisper/__init__.py`
  - Add `file_references_metadata` field to TaskContext
  - Update `BaseLifecycle.pre_execute()` with file reference logic
  - Import new classes from file_references module

- `src/claude_whisper/config.py`
  - Add 3 new configuration fields for file reference feature

- `tests/conftest.py`
  - Add fixtures for testing file reference system

## Testing Strategy

### Unit Tests

**Test FileResolver** (~200 lines):
- test_resolve_main_file_single_match
- test_resolve_main_file_multiple_matches_priority
- test_resolve_config_file
- test_resolve_test_file_with_target
- test_resolve_by_keyword
- test_ignores_hidden_files
- test_scan_directory_caching

**Test FileReferenceDetector** (~150 lines):
- test_detect_main_file
- test_detect_config_file
- test_detect_test_file_with_target
- test_detect_keyword_file
- test_detect_multiple_references
- test_no_overlapping_matches

**Test FileReferenceReplacer** (~150 lines):
- test_replace_single_reference
- test_replace_multiple_references
- test_ambiguity_use_first_strategy
- test_ambiguity_list_all_strategy
- test_no_match_leaves_unchanged

### Integration Tests (~200 lines):
- test_pre_execute_detects_and_replaces
- test_plan_lifecycle_inherits_file_references
- test_end_to_end_single_reference
- test_end_to_end_nested_directories

### Performance Tests (~100 lines):
- test_large_directory_scan (1000 files < 1 second)
- test_caching_effectiveness

## Edge Cases & Error Handling

### Edge Cases Handled

1. **Empty directory**: No matches, references unchanged
2. **Nested directories**: Scans up to configurable depth (default: 3)
3. **Hidden files**: Ignored except known config files (.env)
4. **Symlinks**: Followed during scanning
5. **Case sensitivity**: Pattern matching is case-insensitive
6. **Overlapping patterns**: Higher priority patterns win

### Error Handling

- **Permission errors**: Log warning, continue with partial results
- **Scanning errors**: Log error, cache empty list, don't fail command
- **No matches**: Leave reference unchanged, log warning
- **Invalid regex**: Compile patterns at init, fail fast

## User Feedback

### Desktop Notifications

Enhanced to include file reference info:
```
"Finished task (2 file references resolved)"
```

### Console Logging

```
INFO: Detected 2 file reference(s)
INFO: Replaced 'the config file' with 'src/claude_whisper/config.py'
WARNING: Multiple matches for 'the main file', using highest priority
```

### Bypass Mode Echo (future enhancement)

Show modified command before execution:
```
Original: "update the config file"
Modified: "update src/claude_whisper/config.py"
```

## Performance Considerations

### Optimizations

1. **Directory Scanning**: Results cached per command execution
2. **Max Depth**: Configurable, default 3 levels (prevents deep recursion)
3. **Ignore Patterns**: Skip .git, .venv, __pycache__, node_modules
4. **Lazy Loading**: Only scan if file references detected
5. **Regex Compilation**: Patterns compiled once at initialization

### Expected Performance

- Directory scan (100 files): < 100ms
- Directory scan (1000 files): < 1 second
- Detection + replacement: < 10ms
- Total overhead per command: < 1 second worst case

## Implementation Sequence

### Phase 1: Core Implementation (~400 lines)
1. Create `file_references.py` module
2. Implement `FileResolver` with basic resolvers
3. Implement `FileReferenceDetector` with pattern matching
4. Implement `FileReferenceReplacer` with USE_FIRST strategy
5. Add `DetectedReference` dataclass

### Phase 2: Integration (~50 lines)
1. Add `file_references_metadata` to TaskContext
2. Update `BaseLifecycle.pre_execute()` with detection logic
3. Add configuration fields to Config class
4. Add imports to __init__.py

### Phase 3: Testing (~800 lines)
1. Create comprehensive unit test suite
2. Add integration tests with lifecycle
3. Add performance tests
4. Add edge case tests

### Phase 4: Polish (~50 lines)
1. Add detailed logging throughout
2. Enhance desktop notifications
3. Add documentation strings
4. Handle all edge cases

## Critical Files

Files that will be created or modified:

- **NEW**: `src/claude_whisper/file_references.py` - Core implementation
- **NEW**: `tests/test_file_references.py` - Test suite
- **MODIFY**: `src/claude_whisper/__init__.py` - Integration and TaskContext
- **MODIFY**: `src/claude_whisper/config.py` - Configuration options
- **MODIFY**: `tests/conftest.py` - Test fixtures

## Future Enhancements

### Phase 2 Features (not in this plan)
- Interactive disambiguation (ask user when multiple matches)
- Learning from corrections (track user preferences)
- Context-aware resolution (use git history)

### Phase 3 Features (not in this plan)
- Directory references ("the tests directory")
- Relative references ("the test for this file")
- Fuzzy matching ("confg file" → "config file")
- Language-specific patterns for different project types

## Configuration Examples

### Default (Conservative)
```toml
file_reference_enabled = true
file_reference_strategy = "use_first"
file_reference_max_depth = 3
```

### Aggressive (Show All Options)
```toml
file_reference_enabled = true
file_reference_strategy = "list_all"
file_reference_max_depth = 5
```

### Disabled
```toml
file_reference_enabled = false
```

## Success Criteria

The implementation is successful when:

1. ✅ Natural language file references are detected in commands
2. ✅ References are replaced with actual file paths before Claude sees them
3. ✅ Multiple matches handled gracefully (use highest priority)
4. ✅ No matches leave commands unchanged (fail gracefully)
5. ✅ Comprehensive test coverage (>90%)
6. ✅ Performance overhead < 1 second per command
7. ✅ User feedback via logging and notifications
8. ✅ Configurable via config.toml
9. ✅ Works with both audio and bypass modes
10. ✅ Integrates transparently with existing lifecycle system
