# Plan: Conversation Continuity with Command Similarity Detection

## Overview

Add intelligent conversation continuity to claude-whisper by detecting when new commands are similar to previous conversations and automatically resuming those conversations instead of starting fresh. This leverages the Claude SDK's built-in `resume` parameter to maintain full conversation context.

## Architecture Summary

### Current State
- **Stateless**: Each command creates a new `ClaudeSDKSession` with fresh context
- **No History**: TaskContext only stores metadata, not conversation history
- **SDK Support**: Claude SDK already supports conversation resumption via `resume` parameter in `ClaudeAgentOptions`

### Proposed Solution
1. **Store conversation metadata** (command, session_id, timestamp) in SQLite database
2. **Detect similarity** between incoming commands and recent conversations using hybrid scoring
3. **Resume conversations** automatically by passing session_id to Claude SDK's `resume` parameter
4. **Maintain context** via Claude SDK's built-in transcript management

## Implementation Components

### 1. Conversation Storage (`src/claude_whisper/conversation_store.py`)

**New file** to handle persistent storage of conversation metadata.

```python
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import aiosqlite
from .config import TaskType

@dataclass
class ConversationMetadata:
    """Metadata about a conversation session."""
    conversation_id: str       # UUID for this conversation
    session_id: str           # Claude SDK session ID (for resume)
    command: str              # Original command text
    task_type: TaskType       # PLAN or EDIT
    working_dir: Path         # Working directory
    started_at: datetime      # When conversation started
    last_updated_at: datetime # Last message timestamp
    message_count: int        # Number of turns
    is_active: bool          # True if conversation is still ongoing

class ConversationStore:
    """SQLite-backed storage for conversation metadata."""

    def __init__(self, storage_path: Path):
        """Initialize database at ~/.config/claude-whisper/conversations.db"""
        self.storage_path = storage_path
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables and indexes if they don't exist."""
        # CREATE TABLE conversations (...)
        # CREATE INDEX idx_working_dir, idx_task_type, idx_last_updated

    async def save_conversation(self, metadata: ConversationMetadata):
        """Insert or update conversation metadata."""
        # INSERT OR REPLACE INTO conversations

    async def get_recent_conversations(
        self,
        limit: int = 20,
        task_type: TaskType | None = None,
        working_dir: Path | None = None
    ) -> list[ConversationMetadata]:
        """Get recent active conversations, optionally filtered."""
        # SELECT * FROM conversations WHERE is_active = 1
        # AND task_type = ? AND working_dir = ?
        # ORDER BY last_updated_at DESC LIMIT ?

    async def mark_inactive(self, conversation_id: str):
        """Mark conversation as inactive (ended/timed out)."""
        # UPDATE conversations SET is_active = 0 WHERE conversation_id = ?

    async def get_by_session_id(self, session_id: str) -> ConversationMetadata | None:
        """Retrieve conversation by Claude SDK session ID."""
        # SELECT * FROM conversations WHERE session_id = ?
```

**Database Schema:**
```sql
CREATE TABLE conversations (
    conversation_id TEXT PRIMARY KEY,
    session_id TEXT UNIQUE NOT NULL,
    command TEXT NOT NULL,
    task_type TEXT NOT NULL,
    working_dir TEXT NOT NULL,
    started_at TIMESTAMP NOT NULL,
    last_updated_at TIMESTAMP NOT NULL,
    message_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_working_dir ON conversations(working_dir, is_active);
CREATE INDEX idx_task_type ON conversations(task_type, is_active);
CREATE INDEX idx_last_updated ON conversations(last_updated_at DESC);
```

**Storage Location:** `~/.config/claude-whisper/conversations.db`

### 2. Similarity Detection (`src/claude_whisper/similarity.py`)

**New file** for command similarity detection using hybrid scoring.

```python
import re
from datetime import datetime
from .conversation_store import ConversationMetadata

class CommandSimilarityDetector:
    """Detect similarity between commands using hybrid scoring."""

    def __init__(self):
        # Follow-up phrases that indicate continuation intent
        self.follow_up_phrases = {
            "now", "also", "next", "additionally", "furthermore",
            "continue", "then", "after that", "and", "plus",
            "also add", "also fix", "also update"
        }

        # Common stop words to exclude from keyword matching
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on',
            'at', 'to', 'for', 'with', 'is', 'are', 'was', 'were'
        }

    async def find_similar_conversation(
        self,
        command: str,
        recent_conversations: list[ConversationMetadata],
        threshold: float = 0.7
    ) -> ConversationMetadata | None:
        """Find most similar recent conversation above threshold."""

        if not recent_conversations:
            return None

        command_lower = command.lower()

        # Check for follow-up phrase indicators
        has_follow_up = any(
            phrase in command_lower
            for phrase in self.follow_up_phrases
        )

        # Score all recent conversations
        scores = []
        for conv in recent_conversations:
            score = self._compute_similarity(command, conv.command)

            # Boost score for follow-up phrases (strong continuation signal)
            if has_follow_up:
                score = min(1.0, score + 0.3)

            # Apply temporal decay (prefer recent conversations)
            age_hours = (datetime.now() - conv.last_updated_at).total_seconds() / 3600
            decay_factor = max(0.5, 1 - (age_hours / 24))
            score = score * decay_factor

            scores.append((score, conv))

        # Get best match
        scores.sort(reverse=True, key=lambda x: x[0])
        best_score, best_conv = scores[0]

        if best_score >= threshold:
            return best_conv

        return None

    def _compute_similarity(self, cmd1: str, cmd2: str) -> float:
        """Compute similarity using keyword overlap (Jaccard)."""

        # Tokenize and extract keywords
        tokens1 = set(re.findall(r'\b\w+\b', cmd1.lower()))
        tokens2 = set(re.findall(r'\b\w+\b', cmd2.lower()))

        # Remove stop words
        tokens1 = tokens1 - self.stop_words
        tokens2 = tokens2 - self.stop_words

        if not tokens1 or not tokens2:
            return 0.0

        # Jaccard similarity: |intersection| / |union|
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def _extract_entities(self, command: str) -> set[str]:
        """Extract file paths and code entities from command."""
        entities = set()

        # Extract file paths (e.g., user.py, src/auth/login.js)
        file_patterns = re.findall(r'\b[\w/]+\.\w+\b', command)
        entities.update(file_patterns)

        # Extract function/class names (camelCase, snake_case)
        code_entities = re.findall(r'\b[a-z_][a-z0-9_]*\b', command.lower())
        entities.update(code_entities)

        return entities
```

**Similarity Scoring Strategy:**
1. **Follow-up Phrase Detection** (Weight: +0.3 boost)
   - Phrases like "now", "also", "next" strongly indicate continuation
   - If detected, boost similarity score by 0.3

2. **Keyword Overlap** (Base score: 0-1)
   - Extract keywords, remove stop words
   - Compute Jaccard similarity: intersection/union
   - Extract file paths and code entities for stronger matches

3. **Temporal Decay** (Multiplier: 0.5-1.0)
   - Recent conversations preferred
   - Conversations < 24 hours old: decay_factor = 1 - (age_hours / 24)
   - Minimum decay factor: 0.5

**Default Threshold:** 0.7 (configurable via config)

### 3. Enhanced TaskContext (`src/claude_whisper/__init__.py`)

**Modify existing dataclass** to include conversation continuity fields:

```python
@dataclass
class TaskContext:
    task_id: str
    task_type: TaskType
    command: str
    working_dir: Path
    permission_mode: str
    started_at: datetime = field(default_factory=datetime.now)

    # NEW: Conversation continuity fields
    conversation_id: str | None = None      # UUID for conversation thread
    session_id: str | None = None           # Claude SDK session ID
    is_continuation: bool = False           # True if resuming conversation
    previous_command: str | None = None     # Command from previous turn

    # Existing fields
    output: str | None = None
    finished_at: datetime | None = None
    error: Exception | None = None
```

### 4. Enhanced LifecycleManager (`src/claude_whisper/__init__.py`)

**Modify existing class** to integrate similarity detection:

```python
class LifecycleManager:
    def __init__(self):
        self.detector = TaskTypeDetector()
        # NEW: Initialize conversation store and similarity detector
        self.store = ConversationStore(
            Path.home() / ".config/claude-whisper/conversations.db"
        )
        self.similarity_detector = CommandSimilarityDetector()

    async def create_context(
        self,
        command: str,
        working_dir: Path,
        permission_mode: str
    ) -> TaskContext:
        """Create context, potentially continuing existing conversation."""

        task_type = self.detector.detect(command, permission_mode)

        # Check if conversation continuity is enabled
        if not config.enable_conversation_continuity:
            # Original behavior: always create new conversation
            return TaskContext(
                task_id=str(uuid4()),
                task_type=task_type,
                command=command,
                working_dir=working_dir,
                permission_mode=permission_mode,
            )

        # Find similar recent conversations
        recent_conversations = await self.store.get_recent_conversations(
            limit=20,
            task_type=task_type,
            working_dir=working_dir
        )

        similar_conv = await self.similarity_detector.find_similar_conversation(
            command=command,
            recent_conversations=recent_conversations,
            threshold=config.similarity_threshold
        )

        if similar_conv:
            # Continue existing conversation
            logger.info(
                f"Continuing conversation {similar_conv.conversation_id} "
                f"(similarity score above {config.similarity_threshold})"
            )
            return TaskContext(
                task_id=str(uuid4()),
                task_type=task_type,
                command=command,
                working_dir=working_dir,
                permission_mode=permission_mode,
                conversation_id=similar_conv.conversation_id,
                session_id=similar_conv.session_id,  # KEY: Used for SDK resume
                is_continuation=True,
                previous_command=similar_conv.command,
            )
        else:
            # Start new conversation
            conversation_id = str(uuid4())
            logger.info(f"Starting new conversation {conversation_id}")
            return TaskContext(
                task_id=str(uuid4()),
                task_type=task_type,
                command=command,
                working_dir=working_dir,
                permission_mode=permission_mode,
                conversation_id=conversation_id,
                is_continuation=False,
            )

    # ... rest of class unchanged
```

**Key Change:** `create_context` is now `async` to support database queries.

### 5. Enhanced ClaudeSDKSession (`src/claude_whisper/__init__.py`)

**Modify existing class** to save conversation metadata after execution:

```python
class ClaudeSDKSession:
    """Maintains a single conversation session with Claude."""

    def __init__(self, options: ClaudeAgentOptions, ctx: TaskContext, store: ConversationStore):
        self.client = ClaudeSDKClient(options)
        self.ctx = ctx
        self.lifecycle = lifecycle_manager.get_lifecycle(ctx)
        self.store = store  # NEW: Store reference

    async def run(self, command: str):
        """Execute a command and process the response."""
        try:
            await self.lifecycle.on_start(self.ctx)
            await self.client.connect()

            await self.lifecycle.pre_execute(self.ctx)
            await self.lifecycle.execute(self.ctx, self.client)

            # NEW: Extract session_id from client after execution
            # Note: This assumes the SDK client exposes session_id somehow
            # If not available via client, may need to extract from response
            if hasattr(self.client, 'session_id'):
                self.ctx.session_id = self.client.session_id

            await self.client.disconnect()

            # NEW: Save conversation metadata to database
            if self.ctx.conversation_id:
                metadata = ConversationMetadata(
                    conversation_id=self.ctx.conversation_id,
                    session_id=self.ctx.session_id or str(uuid4()),
                    command=command,
                    task_type=self.ctx.task_type,
                    working_dir=self.ctx.working_dir,
                    started_at=self.ctx.started_at,
                    last_updated_at=datetime.now(),
                    message_count=1 if not self.ctx.is_continuation else 2,
                    is_active=True
                )
                await self.store.save_conversation(metadata)

            await self.lifecycle.on_finish(self.ctx)
        except Exception as e:
            await self.lifecycle.on_error(self.ctx, str(e))
            raise
```

**Key Addition:** Save conversation metadata after successful execution.

### 6. Update _run_claude_task Function (`src/claude_whisper/__init__.py`)

**Modify existing function** to use async create_context and pass resume parameter:

```python
async def _run_claude_task(command: str, working_dir: Path) -> None:
    """Create and run a Claude task with conversation continuity."""

    # NEW: create_context is now async
    ctx = await lifecycle_manager.create_context(
        command, working_dir, config.permission_mode
    )

    permission_mode = config.permission_mode if ctx.task_type == TaskType.EDIT else TaskType.PLAN

    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Bash"],
        permission_mode=permission_mode,
        cwd=working_dir,
        system_prompt={"type": "preset", "preset": "claude_code"},
        setting_sources=["project"],
        # NEW: Use resume parameter to continue conversation
        resume=ctx.session_id if ctx.is_continuation else None,
    )

    # NEW: Pass store to session
    session = ClaudeSDKSession(options, ctx, lifecycle_manager.store)
    await session.run(command)
```

**Key Change:** Pass `resume=ctx.session_id` to `ClaudeAgentOptions` when continuing.

### 7. Configuration Updates (`src/claude_whisper/config.py`)

**Add new configuration fields:**

```python
class Config(BaseSettings):
    # ... existing fields ...

    # NEW: Conversation continuity settings
    enable_conversation_continuity: bool = Field(
        default=True,
        description="Enable automatic conversation continuity detection"
    )

    similarity_threshold: float = Field(
        default=0.7,
        description="Similarity threshold (0-1) for resuming conversations",
        ge=0.0,
        le=1.0
    )

    conversation_timeout_hours: int = Field(
        default=24,
        description="Hours after which conversations are marked inactive"
    )

    max_stored_conversations: int = Field(
        default=100,
        description="Maximum number of conversations to keep in history"
    )
```

### 8. Dependencies (`pyproject.toml`)

**Add new dependency:**

```toml
dependencies = [
    "claude-agent-sdk>=0.1.18",
    "desktop-notifier>=6.2.0",
    "loguru>=0.7.3",
    "mlx-whisper>=0.4.3",
    "pyaudio>=0.2.14",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "pynput>=1.8.1",
    "aiosqlite>=0.19.0",  # NEW: For async SQLite operations
]
```

## User Experience Examples

### Example 1: Continuation Detected
```
User: "fix the authentication bug in user.py"
→ New conversation started (ID: abc123)

User: "now add logging to that function"
→ Detected similarity (score: 0.85)
→ Continuing conversation abc123
→ Claude has full context of previous fix
```

### Example 2: New Topic
```
User: "fix the authentication bug in user.py"
→ New conversation started (ID: abc123)

User: "create a README for the project"
→ Low similarity (score: 0.35)
→ Starting new conversation (ID: def456)
→ Fresh context for unrelated task
```

### Example 3: Follow-up Phrase
```
User: "design a caching system"
→ New PLAN conversation (ID: abc123)

User: "also consider Redis as an option"
→ "also" triggers continuation boost
→ Continuing conversation abc123 with planning context
```

## Implementation Order

### Phase 1: Core Infrastructure
1. Create `src/claude_whisper/conversation_store.py`
   - Implement `ConversationMetadata` dataclass
   - Implement `ConversationStore` class with SQLite operations
   - Add database initialization and table creation

2. Create `src/claude_whisper/similarity.py`
   - Implement `CommandSimilarityDetector` class
   - Implement keyword-based similarity scoring
   - Implement temporal decay and follow-up phrase detection

### Phase 2: Integration
3. Modify `src/claude_whisper/config.py`
   - Add conversation continuity configuration fields

4. Modify `src/claude_whisper/__init__.py`
   - Enhance `TaskContext` dataclass with new fields
   - Update `LifecycleManager.__init__()` to initialize store and detector
   - Convert `LifecycleManager.create_context()` to async
   - Update `ClaudeSDKSession.__init__()` to accept store parameter
   - Update `ClaudeSDKSession.run()` to save conversation metadata
   - Update `_run_claude_task()` to use async create_context and resume

5. Update `pyproject.toml`
   - Add `aiosqlite` dependency

### Phase 3: Testing & Validation
6. Test conversation creation and storage
7. Test similarity detection with various command pairs
8. Test conversation resumption via Claude SDK
9. Test temporal decay and threshold behavior
10. Test working directory and task type filtering

### Phase 4: Cleanup & Documentation
11. Add logging for debugging similarity scores
12. Add error handling for database operations
13. Implement conversation cleanup for old/inactive conversations
14. Update README with new feature documentation

## Critical Implementation Notes

### 1. Session ID Extraction
The Claude SDK's `ClaudeSDKClient` must expose a `session_id` after execution. If this is not directly available:
- **Option A**: Check if it's in the response metadata
- **Option B**: Extract from client's internal state
- **Option C**: Use conversation_id as fallback if session_id unavailable

### 2. Resume Parameter
The `resume` parameter in `ClaudeAgentOptions` must accept the session ID from a previous conversation. Verify this is the correct way to resume conversations in the Claude SDK documentation.

### 3. Backward Compatibility
All changes maintain backward compatibility:
- If `enable_conversation_continuity = False`, system behaves as before
- Database operations fail gracefully (log error, continue with new conversation)
- Similarity detection failure falls back to new conversation

### 4. Performance
- Database queries are indexed for fast lookups (< 10ms)
- Similarity computation is O(n) where n ≤ 20 (recent conversations only)
- Total overhead per command: < 50ms

### 5. Storage Management
- Implement periodic cleanup of old conversations (older than `conversation_timeout_hours`)
- Implement max limit enforcement (keep only `max_stored_conversations` most recent)
- Database should be < 1MB for 100 conversations

## Critical Files to Modify

1. **`/Users/sidhu/claude-whisper/src/claude_whisper/__init__.py`** (418 lines)
   - Lines 39-51: Enhance `TaskContext` dataclass
   - Lines 189-211: Enhance `LifecycleManager` class
   - Lines 214-238: Enhance `ClaudeSDKSession` class
   - Lines 241-255: Update `_run_claude_task` function

2. **`/Users/sidhu/claude-whisper/src/claude_whisper/config.py`** (90 lines)
   - Lines 87+: Add conversation continuity config fields

3. **`/Users/sidhu/claude-whisper/pyproject.toml`** (69 lines)
   - Line 19+: Add `aiosqlite` dependency

4. **`/Users/sidhu/claude-whisper/src/claude_whisper/conversation_store.py`** (NEW)
   - ~200 lines: Complete storage implementation

5. **`/Users/sidhu/claude-whisper/src/claude_whisper/similarity.py`** (NEW)
   - ~150 lines: Complete similarity detection implementation

## Testing Strategy

### Unit Tests
- `test_conversation_store.py`: Database CRUD operations
- `test_similarity.py`: Similarity scoring accuracy
- `test_lifecycle_manager.py`: Context creation with continuity

### Integration Tests
- Test full flow: command → similarity detection → conversation resume
- Test edge cases: empty database, no similar conversations, threshold boundary
- Test working directory isolation (conversations don't leak across projects)

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| SDK `resume` parameter doesn't work as expected | Verify with SDK docs; add fallback to new conversation |
| Database corruption | Add backup/recovery; graceful degradation to stateless mode |
| Similarity false positives | Adjustable threshold; add user override mechanism (future) |
| Storage growth | Automatic cleanup; configurable max limit |
| Performance overhead | Index optimization; limit search to 20 recent conversations |

## Future Enhancements (Out of Scope)

- Manual conversation selection via CLI menu
- Embedding-based similarity (using sentence-transformers)
- Conversation branching (fork conversations)
- Conversation history viewer/manager
- Multi-turn conversation limits
- Conversation export/import
