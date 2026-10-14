"""Pruebas de las herramientas especializadas."""

import json

import pytest

from multi_agent_orchestrator.state import AnalysisArtifact, ResearchArtifact
from multi_agent_orchestrator.tools import (
    calculate_percentage_change,
    raise_if_tool_error,
    search_market_data,
)


@pytest.mark.asyncio
async def test_search_returns_2023_and_2024_market_data() -> None:
    raw = await search_market_data.ainvoke(
        {"query": "mercado energía solar crecimiento 2023 2024", "limit": 4}
    )
    artifact = ResearchArtifact.model_validate_json(raw)
    market = {f.period: f for f in artifact.findings if f.label == "Tamaño del mercado solar"}

    assert market["2023"].value == 120.0
    assert market["2024"].value == 144.0
    assert market["2023"].unit == market["2024"].unit == "millones de USD"
    assert market["2023"].source_id == "market-report-2023"
    assert artifact.confidence >= 0.6


@pytest.mark.asyncio
async def test_search_is_reproducible() -> None:
    query = {"query": "mercado energía solar 2023 2024", "limit": 4}
    assert await search_market_data.ainvoke(query) == await search_market_data.ainvoke(query)


@pytest.mark.asyncio
async def test_search_timeout_returns_controlled_error() -> None:
    raw = await search_market_data.ainvoke({"query": "__simulate_timeout__"})
    result = json.loads(raw)

    assert result["status"] == "error"
    assert result["error_type"] == "TimeoutError"
    with pytest.raises(ValueError, match="TimeoutError"):
        raise_if_tool_error(raw)


@pytest.mark.asyncio
async def test_search_without_matches_returns_not_found() -> None:
    raw = await search_market_data.ainvoke({"query": "xyz qwerty"})

    assert json.loads(raw)["status"] == "not_found"


def test_search_schema_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        search_market_data.args_schema.model_validate({"query": "ab"})  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_growth_tool_calculates_exactly_twenty_percent() -> None:
    # ((144 - 120) / 120) × 100 = 24 / 120 × 100 = 20
    assert ((144 - 120) / 120) * 100 == 20.0
    raw = await calculate_percentage_change.ainvoke(
        {"start_value": 120, "end_value": 144, "start_period": "2023", "end_period": "2024"}
    )
    artifact = AnalysisArtifact.model_validate_json(raw)

    assert artifact.result == 20.0
    assert artifact.start_value == 120.0
    assert artifact.end_value == 144.0
    assert "valor_inicial" in artifact.formula
    assert "+20.00%" in artifact.interpretation


@pytest.mark.asyncio
async def test_growth_tool_handles_zero_base() -> None:
    raw = await calculate_percentage_change.ainvoke(
        {"start_value": 0, "end_value": 10, "start_period": "2023", "end_period": "2024"}
    )
    result = json.loads(raw)

    assert result["status"] == "error"
    assert result["error_type"] == "ZeroDivisionError"
    with pytest.raises(ValueError, match="ZeroDivisionError"):
        raise_if_tool_error(raw)


def test_tools_have_descriptive_docstrings() -> None:
    for item in (search_market_data, calculate_percentage_change):
        assert len(item.description) > 80
