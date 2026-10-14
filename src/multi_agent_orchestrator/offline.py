"""Modelo determinista sin red para la demo offline, el notebook y las pruebas.

Reemplaza a `ChatOpenAI` respetando la misma interfaz que usan los nodos
(`bind_tools()` y `with_structured_output()`), así que el grafo, las herramientas, los
validadores Pydantic y el checkpointer se ejecutan igual que en la demo real. Lo único que
cambia es quién decide: acá son reglas fijas, no un LLM. No consume tokens.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from multi_agent_orchestrator.state import (
    AnalysisArtifact,
    FinalReport,
    ResearchArtifact,
    SupervisorDecision,
)
from multi_agent_orchestrator.supervisor import first_missing_route

OFFLINE_SEARCH_QUERY = "tamaño del mercado de energía solar 2023 2024"


def _payload(messages: Sequence[BaseMessage]) -> dict[str, Any]:
    """Los nodos envían su contexto como JSON en el último HumanMessage."""

    for message in reversed(messages):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            data = json.loads(message.content)
            if isinstance(data, dict):
                return data
    raise ValueError("No se encontró un payload JSON para el modelo offline.")


def supervisor_decision(payload: dict[str, Any]) -> SupervisorDecision:
    route = first_missing_route(
        {
            "research_output": payload.get("research_output"),
            "analysis_output": payload.get("analysis_output"),
            "final_report": payload.get("final_report"),
        }
    )
    reasons = {
        "researcher": "Falta evidencia con valores, períodos, unidades y fuentes.",
        "analyst": "La evidencia cumple la rúbrica; falta calcular la variación.",
        "synthesizer": "Investigación y análisis cumplen la rúbrica; falta el informe.",
        "FINISH": "El informe final cumple la rúbrica y la tarea está completa.",
    }
    return SupervisorDecision(next_agent=route, reason=reasons[route], quality_feedback="")


def research_tool_args(payload: dict[str, Any]) -> dict[str, Any]:
    del payload
    return {"query": OFFLINE_SEARCH_QUERY, "limit": 3}


def analysis_tool_args(payload: dict[str, Any]) -> dict[str, Any]:
    """Elige la primera métrica que aparece en dos períodos con la misma unidad."""

    research = ResearchArtifact.model_validate(payload["research"])
    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for finding in research.findings:
        groups[(finding.label, finding.unit)].append(finding)
    for findings in groups.values():
        if len({item.period for item in findings}) >= 2:
            ordered = sorted(findings, key=lambda item: item.period)
            start, end = ordered[0], ordered[-1]
            return {
                "start_value": start.value,
                "end_value": end.value,
                "start_period": start.period,
                "end_period": end.period,
            }
    raise ValueError("No hay dos períodos comparables en la investigación.")


def final_report(payload: dict[str, Any]) -> FinalReport:
    research = ResearchArtifact.model_validate(payload["research"])
    analysis = AnalysisArtifact.model_validate(payload["analysis"])
    sources = [
        f"{item.label} {item.period}: {item.value:g} {item.unit} [{item.source_id}]"
        for item in sorted(research.findings, key=lambda finding: finding.period)
        if item.value in {analysis.start_value, analysis.end_value}
    ]
    return FinalReport(
        title="Crecimiento del mercado solar 2023-2024",
        executive_summary=(
            f"Según la base simulada, el mercado pasó de {analysis.start_value:g} a "
            f"{analysis.end_value:g} millones de USD: una variación de {analysis.result:+.2f}%."
        ),
        evidence=[*sources, f"Cálculo: {analysis.formula} = {analysis.result:.2f}%"],
        recommendation=(
            "Considerar el segmento solar como prioritario y validar la tendencia con fuentes "
            "de mercado reales antes de comprometer inversión."
        ),
        limitations=[
            "Las cifras provienen de una base de conocimiento simulada.",
            "Dos puntos anuales no alcanzan para proyectar una tendencia.",
        ],
    )


class _ToolCaller:
    def __init__(self, tool_name: str) -> None:
        self._tool_name = tool_name

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        payload = _payload(messages)
        if self._tool_name == "search_market_data":
            args = research_tool_args(payload)
        elif self._tool_name == "calculate_percentage_change":
            args = analysis_tool_args(payload)
        else:
            raise ValueError(f"Herramienta desconocida para el modelo offline: {self._tool_name}")
        return AIMessage(
            content="",
            tool_calls=[{"name": self._tool_name, "args": args, "id": f"call_{uuid4().hex[:8]}"}],
        )


class _StructuredResponder:
    def __init__(self, schema: type[BaseModel]) -> None:
        self._schema = schema

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> BaseModel:
        payload = _payload(messages)
        if self._schema is SupervisorDecision:
            return supervisor_decision(payload)
        if self._schema is FinalReport:
            return final_report(payload)
        raise ValueError(f"Esquema no soportado por el modelo offline: {self._schema.__name__}")


class OfflineDemoModel:
    """Doble determinista de un chat model: misma interfaz, cero llamadas de red."""

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: str | None = None) -> _ToolCaller:
        names = [getattr(item, "name", "") for item in tools]
        return _ToolCaller(tool_choice or names[0])

    def with_structured_output(self, schema: type[BaseModel]) -> _StructuredResponder:
        return _StructuredResponder(schema)
