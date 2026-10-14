"""Grafo: delegación completa, refinamiento, condición de parada y persistencia."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from conftest import analysis_output, final_report, make_state, research_output
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from multi_agent_orchestrator.graph import TeamNodes, build_graph, route_supervisor
from multi_agent_orchestrator.main import build_team
from multi_agent_orchestrator.offline import OfflineDemoModel
from multi_agent_orchestrator.state import AgentState
from multi_agent_orchestrator.supervisor import first_missing_route

EXPECTED_FLOW = [
    "supervisor",
    "researcher",
    "supervisor",
    "analyst",
    "supervisor",
    "synthesizer",
    "supervisor",
]


@pytest.mark.asyncio
async def test_supervisor_refines_insufficient_research_before_delegating() -> None:
    research_attempts = 0

    async def supervisor(state: AgentState) -> dict[str, Any]:
        research = state.get("research_output")
        if research and research["confidence"] < 0.6:
            route, feedback = "researcher", "Confianza insuficiente: reforzá la evidencia."
        else:
            route, feedback = first_missing_route(state), ""
        return {
            "next_agent": route,
            "quality_feedback": feedback,
            "step_count": state.get("step_count", 0) + 1,
            "task_completed": route == "FINISH",
            "contributions": [{"agent": "supervisor", "next_agent": route}],
        }

    async def researcher(state: AgentState) -> dict[str, Any]:
        nonlocal research_attempts
        research_attempts += 1
        if research_attempts == 2:
            assert "reforzá" in state.get("quality_feedback", "")
        confidence = 0.4 if research_attempts == 1 else 0.9
        return {
            "research_output": research_output(confidence),
            "messages": [AIMessage(content="Investigación", name="researcher")],
            "contributions": [{"agent": "researcher", "confidence": confidence}],
        }

    async def analyst(state: AgentState) -> dict[str, Any]:
        del state
        return {"analysis_output": analysis_output(), "contributions": [{"agent": "analyst"}]}

    async def synthesizer(state: AgentState) -> dict[str, Any]:
        del state
        return {"final_report": final_report(), "contributions": [{"agent": "synthesizer"}]}

    graph = build_graph(TeamNodes(supervisor, researcher, analyst, synthesizer))
    result = await graph.ainvoke(make_state(), config={"recursion_limit": 24})
    agents = [item["agent"] for item in result["contributions"]]

    assert research_attempts == 2
    assert agents == ["supervisor", "researcher", *EXPECTED_FLOW]
    assert result["task_completed"] is True


@pytest.mark.asyncio
async def test_graph_stops_a_supervisor_that_delegates_forever() -> None:
    async def endless_supervisor(state: AgentState) -> dict[str, Any]:
        return {"next_agent": "researcher", "step_count": state.get("step_count", 0) + 1}

    async def empty_worker(state: AgentState) -> dict[str, Any]:
        del state
        return {}

    graph = build_graph(
        TeamNodes(endless_supervisor, empty_worker, empty_worker, empty_worker)
    )
    result = await graph.ainvoke(make_state(max_steps=3), config={"recursion_limit": 24})

    assert result["step_count"] == 3
    assert result["task_completed"] is False


def test_router_maps_unknown_route_to_finish() -> None:
    assert route_supervisor(make_state(next_agent="writer")) == "FINISH"
    assert route_supervisor(make_state(next_agent="analyst")) == "analyst"
    assert route_supervisor(make_state(next_agent="analyst", step_count=8)) == "FINISH"


@pytest.mark.asyncio
async def test_real_nodes_visit_every_specialist_and_persist_in_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "checkpoints.sqlite"
    db_path.parent.mkdir(parents=True)
    config: Any = {"configurable": {"thread_id": "test-thread"}, "recursion_limit": 18}

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        graph = build_graph(build_team(OfflineDemoModel()), checkpointer=checkpointer)
        visited = [
            node
            async for update in graph.astream(make_state(), config=config, stream_mode="updates")
            for node in update
        ]

    assert visited == EXPECTED_FLOW

    # Un checkpointer nuevo sobre el mismo archivo recupera el hilo: la persistencia es real.
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        graph = build_graph(build_team(OfflineDemoModel()), checkpointer=checkpointer)
        snapshot = await graph.aget_state(config)

    state = snapshot.values
    assert state["task_completed"] is True
    assert state["analysis_output"]["result"] == 20.0
    assert state["errors"] == []
    assert [c["agent"] for c in state["contributions"]] == EXPECTED_FLOW
