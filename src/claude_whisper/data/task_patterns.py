"""Regex patterns for detecting task types from voice commands."""

PLAN_PATTERNS = [
    r"\bplan\b",
    r"\bdesign\b",
    r"\barchitect(ure)?\b",
    r"\bpropos(e|al)\b",
    r"\bstrategy\b",
    r"\bapproach\s+for\b",
    r"\bblueprint\b",
]

EDIT_PATTERNS = [
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
