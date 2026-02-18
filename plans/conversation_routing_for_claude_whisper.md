# Plan: Conversation Routing for Claude Whisper

## Context

Every voice command currently creates a brand new, isolated Claude session. There is no concept of conversation continuity — if you say "fix the auth bug" and then follow up with "also add a test for that", the second command starts a completely fresh session with no awareness of the first. The Claude Agent SDK (v0.1.31) already supports session resumption via `ClaudeAgentOptions.resume` and returns a `session_id` in `ResultMessage`. This plan adds an in-memory routing layer that tracks recent sessions and decides whether each new command should resume an existing conversation or start fresh.

---

## Files to Create

### `src/claude_whisper/conversation.py` (new)

Contains all routing logic: `ConversationStatus`, `ConversationRecord`, `RoutingDecision`, `ConversationStore`, and `ConversationRouter`.

```python
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class ConversationStatus(str, Enum):
    ACTIVE = "active"      # Currently executing — not routable
    IDLE = "idle"          # Completed, available for resumption
    ERRORED = "errored"    # Failed — not routable


@dataclass
class ConversationRecord:
    session_id: str
    status: ConversationStatus
    created_at: datetime
    last_active_at: datetime
    task_type: str                                          # "plan" or "edit"
    commands: list[str] = field(default_factory=list)       # All commands sent to this session
    topic_keywords: set[str] = field(default_factory=set)   # Extracted significant words
    working_dir: str = ""
    turn_count: int = 0


@dataclass
class RoutingDecision:
    action: str               # "resume" or "new"
    session_id: str | None    # Target session_id if resuming
    confidence: float         # 0.0–1.0
    reason: str               # Human-readable explanation for logging


class ConversationStore:
    """In-memory store tracking active and recent Claude sessions."""

    EXPIRY_WINDOW = timedelta(minutes=30)
    MAX_CONVERSATIONS = 20

    def __init__(self):
        self._conversations: dict[str, ConversationRecord] = {}
        self._lock = threading.Lock()

    def register(self, session_id: str, command: str, task_type: str, working_dir: str) -> ConversationRecord:
        """Register a new conversation after first execution completes."""
        now = datetime.now()
        record = ConversationRecord(
            session_id=session_id,
            status=ConversationStatus.IDLE,
            created_at=now,
            last_active_at=now,
            task_type=task_type,
            commands=[command],
            topic_keywords=self._extract_keywords(command),
            working_dir=working_dir,
            turn_count=1,
        )
        with self._lock:
            self._conversations[session_id] = record
            self._evict_old()
        return record

    def update_after_turn(self, session_id: str, command: str) -> None:
        """Update an existing conversation record after a resumed turn completes."""
        with self._lock:
            record = self._conversations.get(session_id)
            if not record:
                return
            record.commands.append(command)
            record.topic_keywords.update(self._extract_keywords(command))
            record.last_active_at = datetime.now()
            record.turn_count += 1
            record.status = ConversationStatus.IDLE

    def mark_active(self, session_id: str) -> None:
        with self._lock:
            record = self._conversations.get(session_id)
            if record:
                record.status = ConversationStatus.ACTIVE

    def mark_errored(self, session_id: str) -> None:
        with self._lock:
            record = self._conversations.get(session_id)
            if record:
                record.status = ConversationStatus.ERRORED

    def get_routable_conversations(self) -> list[ConversationRecord]:
        """Return IDLE conversations that haven't expired."""
        cutoff = datetime.now() - self.EXPIRY_WINDOW
        with self._lock:
            return [
                r for r in self._conversations.values()
                if r.status == ConversationStatus.IDLE and r.last_active_at > cutoff
            ]

    def get(self, session_id: str) -> ConversationRecord | None:
        with self._lock:
            return self._conversations.get(session_id)

    def _evict_old(self) -> None:
        """Remove expired conversations and enforce max size. Must hold self._lock."""
        cutoff = datetime.now() - self.EXPIRY_WINDOW
        expired = [sid for sid, r in self._conversations.items() if r.last_active_at < cutoff]
        for sid in expired:
            del self._conversations[sid]
        if len(self._conversations) > self.MAX_CONVERSATIONS:
            by_age = sorted(self._conversations.items(), key=lambda x: x[1].last_active_at)
            for sid, _ in by_age[: len(self._conversations) - self.MAX_CONVERSATIONS]:
                del self._conversations[sid]

    @staticmethod
    def _extract_keywords(command: str) -> set[str]:
        """Extract significant words from a command (filters stopwords, keeps nouns/filenames/terms)."""
        STOPWORDS = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "it", "this", "that", "these",
            "those", "i", "you", "he", "she", "we", "they", "me", "him", "her",
            "us", "them", "my", "your", "his", "its", "our", "their", "and",
            "or", "but", "not", "no", "so", "if", "then", "also", "just", "now",
            "please", "make", "go", "get", "same", "too", "very", "really", "about", "into",
        }
        words = set()
        for word in command.lower().split():
            cleaned = word.strip(".,!?;:'\"()-")
            if len(cleaned) > 2 and cleaned not in STOPWORDS:
                words.add(cleaned)
        return words


class ConversationRouter:
    """Scores incoming commands against active conversations to decide resume vs. new."""

    CONTINUATION_PATTERNS = [
        re.compile(p, re.IGNORECASE) for p in [
            r"\balso\b", r"\band\s+then\b", r"\band\s+also\b",
            r"\bcontinue\b", r"\bkeep\s+going\b", r"\bfollow\s*up\b",
            r"\bgoing\s+back\s+to\b", r"\bwhile\s+you'?re\s+at\s+it\b",
            r"\bin\s+that\s+(same\s+)?file\b", r"\bsame\s+thing\b",
            r"\bone\s+more\s+thing\b", r"\bactually\b", r"\bwait\b",
            r"\boh\s+and\b",
        ]
    ]

    KEYWORD_OVERLAP_WEIGHT = 0.4
    RECENCY_WEIGHT = 0.3
    CONTINUATION_SIGNAL_WEIGHT = 0.3
    CONFIDENCE_THRESHOLD = 0.45
    STRONG_RECENCY_WINDOW = timedelta(minutes=3)

    def __init__(self, store: ConversationStore):
        self.store = store

    def route(self, command: str) -> RoutingDecision:
        routable = self.store.get_routable_conversations()
        if not routable:
            return RoutingDecision(action="new", session_id=None, confidence=1.0,
                                   reason="No active conversations to resume")

        has_continuation = self._has_continuation_signal(command)
        command_keywords = ConversationStore._extract_keywords(command)

        scored: list[tuple[ConversationRecord, float, str]] = []
        for record in routable:
            score, reason = self._score(command_keywords, record, has_continuation)
            scored.append((record, score, reason))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_record, best_score, best_reason = scored[0]

        if best_score >= self.CONFIDENCE_THRESHOLD:
            return RoutingDecision(action="resume", session_id=best_record.session_id,
                                   confidence=best_score, reason=best_reason)

        return RoutingDecision(action="new", session_id=None, confidence=1.0 - best_score,
                               reason=f"Best match score {best_score:.2f} below threshold {self.CONFIDENCE_THRESHOLD}")

    def _has_continuation_signal(self, command: str) -> bool:
        return any(p.search(command) for p in self.CONTINUATION_PATTERNS)

    def _score(self, command_keywords: set[str], record: ConversationRecord,
               has_continuation: bool) -> tuple[float, str]:
        reasons = []

        # Keyword overlap (Jaccard similarity)
        if command_keywords and record.topic_keywords:
            intersection = command_keywords & record.topic_keywords
            union = command_keywords | record.topic_keywords
            kw_score = len(intersection) / len(union)
        else:
            kw_score = 0.0
        if kw_score > 0:
            reasons.append(f"keyword overlap={kw_score:.2f}")

        # Recency (exponential decay past 3-min window, half-life 10 min)
        age = (datetime.now() - record.last_active_at).total_seconds()
        strong_secs = self.STRONG_RECENCY_WINDOW.total_seconds()
        if age <= strong_secs:
            recency_score = 1.0
            reasons.append("very recent")
        else:
            recency_score = 0.5 ** ((age - strong_secs) / 600.0)
            reasons.append(f"recency={recency_score:.2f}")

        # Continuation signal (binary)
        cont_score = 1.0 if has_continuation else 0.0
        if has_continuation:
            reasons.append("continuation signal")

        total = (kw_score * self.KEYWORD_OVERLAP_WEIGHT
                 + recency_score * self.RECENCY_WEIGHT
                 + cont_score * self.CONTINUATION_SIGNAL_WEIGHT)

        # Boost: continuation + very recent = strong match
        if has_continuation and age <= strong_secs:
            total = max(total, 0.85)
            reasons.append("strong continuation match")

        return total, "; ".join(reasons) or "no match signals"
```

**Why `task_type: str` instead of `TaskType` enum?** Avoids a circular import between `conversation.py` and `__init__.py`. We pass `ctx.task_type.value` when registering.

---

## Files to Modify

### `src/claude_whisper/__init__.py`

**1. Add import** (line 16 area):
```python
from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage, TextBlock, ToolUseBlock
from .conversation import ConversationStore, ConversationRouter
```

**2. Add fields to `TaskContext`** (after line 93):
```python
session_id: str | None = None           # Captured from ResultMessage after execution
resume_session_id: str | None = None    # Set before execution if resuming an existing session
```

**3. Capture `session_id` in `EditLifecycle.execute`** (after the `for block` loop, still inside the `async for message` loop):
```python
elif isinstance(message, ResultMessage):
    ctx.session_id = message.session_id
```

**4. Capture `session_id` in `PlanLifecycle.execute`** (same pattern as above).

**5. Add module-level singletons** (after `lifecycle_manager = LifecycleManager()` at line 258):
```python
conversation_store = ConversationStore()
conversation_router = ConversationRouter(conversation_store)
```

**6. Update `BaseLifecycle.on_start`** to indicate resumption in the notification:
```python
async def on_start(self, ctx: TaskContext):
    message_preview = ctx.command[0:50] + "..."
    if ctx.resume_session_id:
        await notifier.send(title="Resuming conversation", message=f"Continuing: {message_preview}")
    else:
        await notifier.send(title="Task started", message=f"Started task {message_preview}")
```

**7. Rewrite `ClaudeSDKSession.run`** to register/update the conversation store:
```python
async def run(self, command: str):
    try:
        await self.lifecycle.on_start(self.ctx)
        await self.client.connect()
        await self.lifecycle.pre_execute(self.ctx)
        await self.lifecycle.execute(self.ctx, self.client)
        await self.client.disconnect()
        await self.lifecycle.on_finish(self.ctx)

        # Register or update conversation in the store
        if self.ctx.session_id:
            if self.ctx.resume_session_id:
                conversation_store.update_after_turn(self.ctx.session_id, command)
            else:
                conversation_store.register(
                    session_id=self.ctx.session_id,
                    command=command,
                    task_type=self.ctx.task_type.value,
                    working_dir=str(self.ctx.working_dir),
                )
    except Exception as e:
        if self.ctx.session_id:
            conversation_store.mark_errored(self.ctx.session_id)
        elif self.ctx.resume_session_id:
            conversation_store.mark_errored(self.ctx.resume_session_id)
        raise e
```

**8. Rewrite `_run_claude_task`** to route commands:
```python
async def _run_claude_task(command: str) -> None:
    working_dir = config.cwd

    # Route: resume existing conversation or start new?
    decision = conversation_router.route(command)
    logger.info(f"Routing: {decision.action} (confidence={decision.confidence:.2f}, reason={decision.reason})")

    ctx = lifecycle_manager.create_context(command, working_dir, config.permission_mode)
    logger.debug(f"Starting task with context: {ctx}")

    permission_mode = config.permission_mode if ctx.task_type == TaskType.EDIT else TaskType.PLAN
    options = ClaudeAgentOptions(
        allowed_tools=["mcp__utils__screenshot"],
        disallowed_tools=["AskUserQuestion"],
        permission_mode=permission_mode,
        cwd=working_dir,
        system_prompt={"type": "preset", "preset": "claude_code"},
        setting_sources=["project"],
        mcp_servers={"utils": claude_whisper_mcp_server},
        max_buffer_size=5 * 1024 * 1024,
    )

    # If resuming, set the SDK resume option
    if decision.action == "resume" and decision.session_id:
        options.resume = decision.session_id
        ctx.resume_session_id = decision.session_id
        conversation_store.mark_active(decision.session_id)
        logger.info(f"Resuming session {decision.session_id}")

    session = ClaudeSDKSession(options, ctx)
    await session.run(command)
```

### `src/claude_whisper/config.py` (optional)

Add configurable routing parameters:
```python
routing_enabled: bool = Field(default=True, description="Enable conversation routing")
routing_confidence_threshold: float = Field(default=0.45, description="Min confidence to resume (0.0-1.0)")
routing_expiry_minutes: int = Field(default=30, description="Minutes before a conversation expires from routing")
```

Wire into `ConversationStore.EXPIRY_WINDOW` and `ConversationRouter.CONFIDENCE_THRESHOLD` constructors.

---

## Routing Algorithm Summary

| Signal | Weight | How it works |
|---|---|---|
| Keyword overlap | 0.4 | Jaccard similarity between command keywords and conversation's accumulated keywords |
| Recency | 0.3 | Full score within 3 min, exponential decay after (half-life 10 min) |
| Continuation signal | 0.3 | Presence of phrases like "also", "continue", "same file", "one more thing" |

**Threshold**: 0.45 minimum to resume. The system defaults to **new conversations** unless there's meaningful evidence of continuity. Examples:

- "also add a test for that" 3 seconds after previous → continuation(0.3) + recency(0.3) = 0.6, boosted to **0.85 → RESUME**
- "fix the auth bug" with keyword match 2 min after → keyword(~0.16) + recency(0.3) = **0.46 → RESUME**
- "refactor the database" with no keyword overlap, 5 min later → recency(~0.23) = **0.23 → NEW**
- Empty/short transcription → no keywords, no continuation → **NEW** (safe default)

---

## Edge Cases

- **Concurrent commands**: Active conversations (`status=ACTIVE`) are excluded from routing, preventing two commands from hitting the same session
- **Errored sessions**: Marked `ERRORED`, removed from routing candidates
- **Plan→Edit transitions**: Intentionally allowed — user might plan then say "now implement it" in the same conversation
- **30-min expiry**: Conversations older than 30 minutes are evicted automatically
- **Max 20 conversations**: Oldest evicted when limit is exceeded

---

## Verification

1. **Unit tests** for `ConversationStore`: register, update, mark_active, mark_errored, eviction, keyword extraction
2. **Unit tests** for `ConversationRouter`: scoring logic, continuation detection, threshold behavior, empty store case
3. **Integration test**: simulate sequence of commands and verify routing decisions (new → resume → new for unrelated topic)
4. **Manual test**: run `claude-whisper`, speak a command, speak a follow-up with "also" or related keywords, verify the notification says "Resuming conversation" instead of "Task started"
