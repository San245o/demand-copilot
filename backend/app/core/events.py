from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """The vocabulary of things the UI can render as the crew works."""

    RUN_START = "run_start"
    AGENT_START = "agent_start"  # an agent/node began
    THOUGHT = "thought"  # a ReAct reasoning step
    TOOL_CALL = "tool_call"  # agent invoked a tool
    TOOL_RESULT = "tool_result"  # tool returned
    AGENT_END = "agent_end"  # an agent/node finished
    FORECAST = "forecast"  # structured forecast payload
    INTERRUPT = "interrupt"  # paused for human approval
    BRIEF = "brief"  # final narrative brief
    RUN_END = "run_end"
    ERROR = "error"


class StreamEvent(BaseModel):
    """A single SSE message. `agent` names the node; `data` is type-specific."""

    type: EventType
    agent: str | None = None
    message: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    def to_sse(self) -> dict[str, str]:
        """Shape expected by sse-starlette's EventSourceResponse."""
        return {"event": self.type.value, "data": self.json_data()}

    def json_data(self) -> str:
        return json.dumps(
            {"agent": self.agent, "message": self.message, "data": self.data},
            default=str,
        )
