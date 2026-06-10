from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.nodes import (
    approval_node,
    forecast_node,
    planning_node,
    sensing_node,
    validation_node,
)
from app.agents.state import CrewState


@lru_cache(maxsize=1)
def build_graph():
    """Compile the crew once. MemorySaver checkpointer enables interrupt/resume.

    sensing → forecast → validation → approval (HITL) → planning
    """
    g = StateGraph(CrewState)
    g.add_node("sensing", sensing_node)
    g.add_node("forecast", forecast_node)
    g.add_node("validation", validation_node)
    g.add_node("approval", approval_node)
    g.add_node("planning", planning_node)

    g.add_edge(START, "sensing")
    g.add_edge("sensing", "forecast")
    g.add_edge("forecast", "validation")
    g.add_edge("validation", "approval")
    g.add_edge("approval", "planning")
    g.add_edge("planning", END)

    return g.compile(checkpointer=MemorySaver())
