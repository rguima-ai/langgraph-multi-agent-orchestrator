"""Fixtures compartidas: estados y artefactos válidos, y bloqueo de red."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest

from multi_agent_orchestrator.main import initial_state
from multi_agent_orchestrator.state import AgentState

REQUEST = "Analizá el mercado solar entre 2023 y 2024"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ninguna prueba puede abrir conexiones: si algo intenta llamar a OpenAI, falla."""

    def guard(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Las pruebas no deben usar la red.")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    yield


def make_state(max_steps: int = 8, **overrides: Any) -> AgentState:
    state = initial_state(REQUEST, max_steps)
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def research_output(confidence: float = 0.9) -> dict[str, Any]:
    return {
        "query": "mercado solar 2023 2024",
        "findings": [
            {
                "label": "Tamaño del mercado solar",
                "value": 120.0,
                "unit": "millones de USD",
                "period": "2023",
                "source_id": "market-report-2023",
                "note": "",
            },
            {
                "label": "Tamaño del mercado solar",
                "value": 144.0,
                "unit": "millones de USD",
                "period": "2024",
                "source_id": "market-report-2024",
                "note": "",
            },
        ],
        "summary": "Dos métricas comparables con fuente identificada.",
        "confidence": confidence,
    }


def analysis_output() -> dict[str, Any]:
    return {
        "metric": "Variación porcentual 2023-2024",
        "start_value": 120.0,
        "end_value": 144.0,
        "result": 20.0,
        "formula": "((valor_final - valor_inicial) / valor_inicial) * 100",
        "interpretation": "El indicador cambió +20.00% entre 2023 y 2024.",
        "confidence": 1.0,
    }


def final_report() -> dict[str, Any]:
    return {
        "title": "Informe del mercado solar",
        "executive_summary": "El mercado simulado creció 20% y presenta una señal positiva.",
        "evidence": ["market-report-2023: 120", "market-report-2024: 144"],
        "recommendation": "Continuar el análisis con fuentes reales antes de invertir.",
        "limitations": ["Los datos utilizados son simulados."],
    }
