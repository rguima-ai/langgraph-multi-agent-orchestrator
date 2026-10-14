"""Pruebas del ruteo jerárquico, refinamiento y condición de parada."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from multi_agent_orchestrator.graph import TeamNodes, build_graph
from multi_agent_orchestrator.state import AgentState
from multi_agent_orchestrator.supervisor import SupervisorNode


def _initial_state(max_steps: int = 8) -> AgentState:
    return {
        "messages": [HumanMessage(content="Analizá el mercado solar")],
        "request": "Analizá el mercado solar",
        "research_output": None,
        "analysis_output": None,
        "final_report": None,
        "task_completed": False,
        "step_count": 0,
        "max_steps": max_steps,
        "contributions": [],
        "errors": [],
    }


@pytest.mark.asyncio
async def test_supervisor_refines_then_delegates_and_finishes() -> None:
    research_attempts = 0

    async def supervisor(state: AgentState) -> dict[str, Any]:
        step = state.get("step_count", 0) + 1
        research = state.get("research_output")
        if not research or research["confidence"] < 0.6:
            route = "researcher"
        elif not state.get("analysis_output"):
            route = "analyst"
        elif not state.get("final_report"):
            route = "synthesizer"
        else:
            route = "FINISH"
        return {
            "next_agent": route,
            "step_count": step,
            "task_completed": route == "FINISH",
            "contributions": [
                {"agent": "supervisor", "status": "decision", "next_agent": route}
            ],
        }

    async def researcher(state: AgentState) -> dict[str, Any]:
        del state
        nonlocal research_attempts
        research_attempts += 1
        confidence = 0.4 if research_attempts == 1 else 0.9
        return {
            "research_output": {
                "query": "mercado solar",
                "findings": [
                    {
                        "label": "Mercado",
                        "value": 120,
                        "unit": "USD",
                        "period": "2023",
                        "source_id": "source-2023",
                        "note": "",
                    }
                ],
                "summary": "Investigación simulada para probar refinamiento.",
                "confidence": confidence,
            },
            "messages": [AIMessage(content="Investigación", name="researcher")],
            "contributions": [
                {"agent": "researcher", "status": "success", "confidence": confidence}
            ],
        }

    async def analyst(state: AgentState) -> dict[str, Any]:
        del state
        return {
            "analysis_output": {
                "metric": "Variación",
                "start_value": 120,
                "end_value": 144,
                "result": 20,
                "formula": "(144 - 120) / 120 * 100",
                "interpretation": "El mercado aumentó veinte por ciento.",
                "confidence": 1.0,
            },
            "contributions": [{"agent": "analyst", "status": "success"}],
        }

    async def synthesizer(state: AgentState) -> dict[str, Any]:
        del state
        return {
            "final_report": {
                "title": "Informe del mercado solar",
                "executive_summary": (
                    "El mercado simulado creció y presenta una señal positiva para inversión."
                ),
                "evidence": ["source-2023", "source-2024"],
                "recommendation": "Continuar el análisis con fuentes reales antes de invertir.",
                "limitations": ["Los datos utilizados son simulados."],
            },
            "contributions": [{"agent": "synthesizer", "status": "success"}],
        }

    graph = build_graph(
        TeamNodes(
            supervisor=supervisor,
            researcher=researcher,
            analyst=analyst,
            synthesizer=synthesizer,
        )
    )
    result = await graph.ainvoke(_initial_state(), config={"recursion_limit": 24})
    agents = [item["agent"] for item in result["contributions"]]

    assert research_attempts == 2
    assert agents == [
        "supervisor",
        "researcher",
        "supervisor",
        "researcher",
        "supervisor",
        "analyst",
        "supervisor",
        "synthesizer",
        "supervisor",
    ]
    assert result["task_completed"] is True
    assert result["final_report"]["title"] == "Informe del mercado solar"


@pytest.mark.asyncio
async def test_graph_stops_at_max_supervisor_steps() -> None:
    async def endless_supervisor(state: AgentState) -> dict[str, Any]:
        return {
            "next_agent": "researcher",
            "step_count": state.get("step_count", 0) + 1,
        }

    async def empty_worker(state: AgentState) -> dict[str, Any]:
        del state
        return {}

    graph = build_graph(
        TeamNodes(
            supervisor=endless_supervisor,
            researcher=empty_worker,
            analyst=empty_worker,
            synthesizer=empty_worker,
        )
    )
    result = await graph.ainvoke(_initial_state(max_steps=3), config={"recursion_limit": 24})

    assert result["step_count"] == 3
    assert result["task_completed"] is False


@pytest.mark.asyncio
async def test_supervisor_uses_safe_route_after_recoverable_llm_error() -> None:
    class TimeoutModel:
        def with_structured_output(self, schema: Any) -> "TimeoutModel":
            del schema
            return self

        async def ainvoke(self, messages: Any) -> Any:
            del messages
            raise TimeoutError("simulated timeout")

    update = await SupervisorNode(TimeoutModel())(_initial_state())

    assert update["next_agent"] == "researcher"
    assert update["step_count"] == 1
    assert "TimeoutError" in update["errors"][0]
