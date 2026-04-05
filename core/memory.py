from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import List, Deque

from config import cfg


@dataclass
class Turn:
    role: str   # "user" or "assistant"
    content: str


class ConversationMemory:
    def __init__(self, window: int = cfg.memory_window):
        self._window = window
        self._turns: Deque[Turn] = deque()

    def add(self, role: str, content: str) -> None:
        self._turns.append(Turn(role=role, content=content))
        # keep last window * 2 messages (window pairs)
        while len(self._turns) > self._window * 2:
            self._turns.popleft()

    def to_messages(self) -> List[dict]:
        return [{"role": t.role, "content": t.content} for t in self._turns]

    def format_for_prompt(self) -> str:
        if not self._turns:
            return ""
        lines = ["## Conversation History"]
        for t in self._turns:
            lines.append(f"**{t.role.capitalize()}**: {t.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._turns.clear()

    @property
    def num_turns(self) -> int:
        return len(self._turns)
