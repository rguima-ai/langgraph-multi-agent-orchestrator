"""Supervisor: fallback ante errores, guardas de ruteo y límite de pasos."""

from __future__ import annotations

from typing import Any

import pytest
from conftest import analysis_output, final_report, make_state, research_output
from langchain_core.exceptions import OutputParserException

from multi_agent_orchestrator.state import SupervisorDecision
from multi_agent_orchestrator.supervisor import SupervisorNode


class ScriptedSupervisorModel:
    """Doble mínimo de `with_structured_output`: devuelve o lanza lo programado."""

    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.calls = 0

    def with_structured_output(self, schema: Any) -> ScriptedSupervisorModel:
        assert schema is SupervisorDecision
        return self

    async def ainvoke(self, messages: Any) -> Any:
        del messages
        self.calls += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def decision(route: str, feedback: str = "") -> SupervisorDecision:
    return SupervisorDecision.model_validate(
        {"next_agent": route, "reason": "Decisión programada.", "quality_feedback": feedback}
    )


@pytest.mark.asyncio
async def test_supervisor_uses_safe_route_after_recoverable_timeout() -> None:
    update = await SupervisorNode(ScriptedSupervisorModel(TimeoutError("timeout")))(make_state())

    assert update["next_agent"] == "researcher"
    assert update["step_count"] == 1
    assert "TimeoutError" in update["errors"][0]


@pytest.mark.asyncio
async def test_supervisor_fallback_continues_from_first_missing_artifact() -> None:
    model = ScriptedSupervisorModel(OutputParserException("JSON inválido"))
    state = make_state(research_output=research_output())

    update = await SupervisorNode(model)(state)

    assert update["next_agent"] == "analyst"
    assert "OutputParserException" in update["errors"][0]


@pytest.mark.asyncio
async def test_supervisor_cannot_finish_without_final_report() -> None:
    state = make_state(research_output=research_output(), analysis_output=analysis_output())

    update = await SupervisorNode(ScriptedSupervisorModel(decision("FINISH")))(state)

    assert update["next_agent"] == "synthesizer"
    assert update["task_completed"] is False
    assert "final_report" in update["errors"][0]


@pytest.mark.asyncio
async def test_supervisor_does_not_delegate_without_prerequisites() -> None:
    update = await SupervisorNode(ScriptedSupervisorModel(decision("synthesizer")))(make_state())

    assert update["next_agent"] == "researcher"


@pytest.mark.asyncio
async def test_supervisor_finishes_when_report_exists() -> None:
    state = make_state(
        research_output=research_output(),
        analysis_output=analysis_output(),
        final_report=final_report(),
    )

    update = await SupervisorNode(ScriptedSupervisorModel(decision("FINISH")))(state)

    assert update["next_agent"] == "FINISH"
    assert update["task_completed"] is True
    assert "errors" not in update


@pytest.mark.asyncio
async def test_refining_research_invalidates_downstream_artifacts() -> None:
    state = make_state(
        research_output=research_output(),
        analysis_output=analysis_output(),
        final_report=final_report(),
    )
    model = ScriptedSupervisorModel(decision("researcher", "Agregá una fuente más."))

    update = await SupervisorNode(model)(state)

    assert update["next_agent"] == "researcher"
    assert update["quality_feedback"] == "Agregá una fuente más."
    assert update["analysis_output"] is None
    assert update["final_report"] is None
    assert "research_output" not in update  # se conserva hasta que llegue la versión nueva


@pytest.mark.asyncio
async def test_last_allowed_decision_stops_instead_of_opening_another_cycle() -> None:
    model = ScriptedSupervisorModel(decision("researcher"))
    update = await SupervisorNode(model)(make_state(max_steps=4, step_count=3))

    assert update["next_agent"] == "FINISH"
    assert update["step_count"] == 4
    assert update["task_completed"] is False
    assert "límite" in update["errors"][-1]


@pytest.mark.asyncio
async def test_supervisor_at_limit_does_not_call_the_llm() -> None:
    model = ScriptedSupervisorModel(decision("researcher"))
    update = await SupervisorNode(model)(make_state(max_steps=4, step_count=4))

    assert model.calls == 0
    assert update["next_agent"] == "FINISH"
