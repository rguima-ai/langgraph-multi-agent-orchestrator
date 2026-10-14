"""Pruebas de contratos Pydantic y herramientas especializadas."""

import json

import pytest
from pydantic import ValidationError

from multi_agent_orchestrator.state import ResearchArtifact
from multi_agent_orchestrator.tools import (
    calculate_percentage_change,
    search_market_data,
)


def test_research_artifact_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        ResearchArtifact(
            query="energía solar",
            findings=[],
            summary="Resumen suficientemente descriptivo.",
            confidence=1.5,
        )


@pytest.mark.asyncio
async def test_search_returns_two_solar_periods() -> None:
    raw = await search_market_data.ainvoke(
        {"query": "mercado energía solar crecimiento 2023 2024", "limit": 4}
    )
    artifact = ResearchArtifact.model_validate_json(raw)
    periods = {finding.period for finding in artifact.findings}

    assert {"2023", "2024"}.issubset(periods)
    assert artifact.confidence >= 0.6


@pytest.mark.asyncio
async def test_growth_tool_calculates_twenty_percent() -> None:
    raw = await calculate_percentage_change.ainvoke(
        {
            "start_value": 120,
            "end_value": 144,
            "start_period": "2023",
            "end_period": "2024",
        }
    )
    result = json.loads(raw)

    assert result["result"] == 20.0
    assert result["confidence"] == 1.0


@pytest.mark.asyncio
async def test_growth_tool_handles_zero_base() -> None:
    raw = await calculate_percentage_change.ainvoke(
        {
            "start_value": 0,
            "end_value": 10,
            "start_period": "2023",
            "end_period": "2024",
        }
    )
    result = json.loads(raw)

    assert result["status"] == "error"
    assert result["error_type"] == "ZeroDivisionError"
