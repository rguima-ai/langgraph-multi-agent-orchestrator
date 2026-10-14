"""Especialistas: escriben solo su parcela, validan y registran errores recuperables."""

from __future__ import annotations

from typing import Any

import pytest
from conftest import analysis_output, final_report, make_state, research_output
from langchain_core.messages import AIMessage

from multi_agent_orchestrator.agents import AnalystAgent, ResearchAgent, SynthesisAgent
from multi_agent_orchestrator.state import FinalReport


class ToolCallModel:
    """Doble de `bind_tools`: devuelve una llamada a herramienta con argumentos fijos."""

    def __init__(self, args: dict[str, Any] | None) -> None:
        self.args = args
        self.bound_tools: list[str] = []
        self.tool_choice: str | None = None

    def bind_tools(self, tools: list[Any], *, tool_choice: str) -> ToolCallModel:
        self.bound_tools = [item.name for item in tools]
        self.tool_choice = tool_choice
        return self

    async def ainvoke(self, messages: Any) -> AIMessage:
        del messages
        if self.args is None:
            return AIMessage(content="Respuesta sin herramienta.")
        assert self.tool_choice is not None
        return AIMessage(
            content="",
            tool_calls=[{"name": self.tool_choice, "args": self.args, "id": "call_1"}],
        )


class ReportModel:
    """Doble de `with_structured_output`; no ofrece herramientas a Síntesis."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report

    def with_structured_output(self, schema: Any) -> ReportModel:
        assert schema is FinalReport
        return self

    async def ainvoke(self, messages: Any) -> FinalReport:
        del messages
        return FinalReport.model_validate(self.report)


OWNED_FIELDS = {"research_output", "analysis_output", "final_report"}


def assert_writes_only(update: dict[str, Any], field: str | None) -> None:
    assert OWNED_FIELDS & update.keys() == ({field} if field else set())
    assert "next_agent" not in update
    assert "step_count" not in update


@pytest.mark.asyncio
async def test_researcher_uses_search_tool_and_writes_research_only() -> None:
    model = ToolCallModel({"query": "mercado energía solar 2023 2024"})
    update = await ResearchAgent(model)(make_state())

    assert model.bound_tools == ["search_market_data"]
    assert_writes_only(update, "research_output")
    findings = update["research_output"]["findings"]
    assert all(item["source_id"] and item["unit"] and item["period"] for item in findings)
    assert update["contributions"][0]["status"] == "success"
    assert "market-report-2023" in update["contributions"][0]["source_ids"]


@pytest.mark.asyncio
async def test_researcher_records_search_timeout_without_crashing() -> None:
    update = await ResearchAgent(ToolCallModel({"query": "__simulate_timeout__"}))(make_state())

    assert_writes_only(update, None)
    assert "TimeoutError" in update["errors"][0]
    assert update["contributions"][0]["status"] == "error"


@pytest.mark.asyncio
async def test_researcher_requires_a_tool_call() -> None:
    update = await ResearchAgent(ToolCallModel(None))(make_state())

    assert "herramienta" in update["errors"][0]


@pytest.mark.asyncio
async def test_analyst_calculates_growth_from_researched_values() -> None:
    model = ToolCallModel(
        {"start_value": 120, "end_value": 144, "start_period": "2023", "end_period": "2024"}
    )
    update = await AnalystAgent(model)(make_state(research_output=research_output()))

    assert model.bound_tools == ["calculate_percentage_change"]
    assert_writes_only(update, "analysis_output")
    assert update["analysis_output"]["result"] == 20.0
    assert update["analysis_output"]["formula"]
    assert update["contributions"][0]["source_ids"] == [
        "market-report-2023",
        "market-report-2024",
    ]


@pytest.mark.asyncio
async def test_analyst_rejects_values_not_present_in_research() -> None:
    model = ToolCallModel(
        {"start_value": 100, "end_value": 144, "start_period": "2023", "end_period": "2024"}
    )
    update = await AnalystAgent(model)(make_state(research_output=research_output()))

    assert_writes_only(update, None)
    assert "no aparece en la investigación" in update["errors"][0]


@pytest.mark.asyncio
async def test_analyst_without_research_records_error() -> None:
    update = await AnalystAgent(ToolCallModel({}))(make_state())

    assert_writes_only(update, None)
    assert update["errors"][0].startswith("Analista: ValidationError")


@pytest.mark.asyncio
async def test_synthesizer_writes_report_citing_sources() -> None:
    state = make_state(research_output=research_output(), analysis_output=analysis_output())
    update = await SynthesisAgent(ReportModel(final_report()))(state)

    assert_writes_only(update, "final_report")
    assert update["final_report"]["recommendation"]
    assert update["final_report"]["limitations"]
    assert update["contributions"][0]["source_ids"] == [
        "market-report-2023",
        "market-report-2024",
    ]


@pytest.mark.asyncio
async def test_synthesizer_rejects_report_with_invented_sources() -> None:
    report = {**final_report(), "evidence": ["fuente-inventada-1", "fuente-inventada-2"]}
    state = make_state(research_output=research_output(), analysis_output=analysis_output())

    update = await SynthesisAgent(ReportModel(report))(state)

    assert_writes_only(update, None)
    assert "source_id" in update["errors"][0]


@pytest.mark.asyncio
async def test_synthesizer_requires_research_and_analysis() -> None:
    update = await SynthesisAgent(ReportModel(final_report()))(make_state())

    assert_writes_only(update, None)
    assert update["errors"]
