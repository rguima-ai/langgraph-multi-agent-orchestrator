"""Especialista que calcula métricas a partir de la evidencia investigada."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from multi_agent_orchestrator.errors import RECOVERABLE_LLM_ERRORS
from multi_agent_orchestrator.state import (
    AgentState,
    AnalysisArtifact,
    ResearchArtifact,
    ResearchFinding,
)
from multi_agent_orchestrator.tools import (
    ANALYSIS_TOOLS,
    calculate_percentage_change,
    raise_if_tool_error,
)

ANALYSIS_PROMPT = SystemMessage(
    content=(
        "Sos el agente Analista. Recibís evidencia ya validada. Elegí dos métricas comparables "
        "con la misma unidad y períodos distintos, y usá calculate_percentage_change con "
        "exactamente esos valores y períodos. "
        "No busques nuevas fuentes ni redactes el informe final."
    )
)


def _grounded_source_ids(research: ResearchArtifact, args: dict[str, Any]) -> list[str]:
    """Verifica que los valores elegidos por el LLM existan en la evidencia investigada.

    Sin este control, el modelo podría pasarle a la herramienta números inventados y el
    cálculo, aunque correcto, no estaría basado en las fuentes.
    """

    def find(value_key: str, period_key: str) -> ResearchFinding:
        for finding in research.findings:
            if finding.value == float(args[value_key]) and finding.period == args[period_key]:
                return finding
        raise ValueError(
            f"{value_key}={args[value_key]} ({args[period_key]}) no aparece en la investigación."
        )

    start = find("start_value", "start_period")
    end = find("end_value", "end_period")
    if start.unit != end.unit:
        raise ValueError(f"Unidades no comparables: '{start.unit}' y '{end.unit}'.")
    return [start.source_id, end.source_id]


class AnalystAgent:
    """Nodo async que solo escribe `analysis_output`."""

    def __init__(self, model: Any) -> None:
        self._model = model.bind_tools(
            ANALYSIS_TOOLS,
            tool_choice=calculate_percentage_change.name,
        )

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        try:
            research = ResearchArtifact.model_validate(state.get("research_output"))
            payload = {
                "request": state["request"],
                "research": research.model_dump(mode="json"),
                "supervisor_feedback": state.get("quality_feedback", ""),
            }
            response = await self._model.ainvoke(
                [ANALYSIS_PROMPT, HumanMessage(content=json.dumps(payload, ensure_ascii=False))]
            )
            if not response.tool_calls:
                raise ValueError("El Analista no generó la llamada obligatoria a su herramienta.")

            tool_call = response.tool_calls[0]
            source_ids = _grounded_source_ids(research, tool_call["args"])
            raw_result = await calculate_percentage_change.ainvoke(tool_call["args"])
            raise_if_tool_error(raw_result)
            artifact = AnalysisArtifact.model_validate_json(raw_result)
            summary = (
                f"Análisis validado: {artifact.metric}; resultado {artifact.result:+.2f}%."
            )
            return {
                "analysis_output": artifact.model_dump(mode="json"),
                "messages": [AIMessage(content=summary, name="analyst")],
                "contributions": [
                    {
                        "agent": "analyst",
                        "status": "success",
                        "summary": summary,
                        "source_ids": source_ids,
                    }
                ],
            }
        except (
            ValidationError,
            ValueError,
            KeyError,
            TypeError,
            *RECOVERABLE_LLM_ERRORS,
        ) as exc:
            error = f"Analista: {type(exc).__name__}: {exc}"
            return {
                "messages": [AIMessage(content=error, name="analyst")],
                "errors": [error],
                "contributions": [
                    {"agent": "analyst", "status": "error", "summary": error}
                ],
            }
