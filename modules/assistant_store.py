"""Persistent, linear assistant context (file-backed).

Saves the assistant conversation so it is remembered across app sessions and
kept in strict chronological order (the "linear" thread the model sees).
The file is gitignored; only the most recent ``max_messages`` are kept.
"""
import json
from pathlib import Path

CONTEXT_PATH = Path(__file__).resolve().parent.parent / "assistant_context.json"


def load_context(max_messages=40):
    try:
        data = json.loads(CONTEXT_PATH.read_text(encoding="utf-8"))
        msgs = data.get("messages", [])
        if max_messages:
            msgs = msgs[-int(max_messages):]
        return msgs
    except Exception:  # noqa: BLE001
        return []


def save_context(messages, max_messages=40):
    msgs = list(messages)
    if max_messages:
        msgs = msgs[-int(max_messages):]
    try:
        CONTEXT_PATH.write_text(
            json.dumps({"messages": msgs}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return True
    except Exception:  # noqa: BLE001
        return False


def clear_context():
    try:
        CONTEXT_PATH.unlink()
    except FileNotFoundError:
        pass