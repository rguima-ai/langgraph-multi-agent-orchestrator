"""Especialista que calcula métricas a partir de la evidencia investigada."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from multi_agent_orchestrator.errors import RECOVERABLE_LLM_ERRORS
from multi_agent_orchestrator.state import AgentState, AnalysisArtifact, ResearchArtifact
from multi_agent_orchestrator.tools import ANALYSIS_TOOLS, calculate_percentage_change

ANALYSIS_PROMPT = SystemMessage(
    content=(
        "Sos el agente Analista. Recibís evidencia ya validada. Elegí dos métricas comparables "
        "con la misma unidad y períodos distintos, y usá calculate_percentage_change. "
        "No busques nuevas fuentes ni redactes el informe final."
    )
)


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
            instruction = (
                f"Solicitud original: {state['request']}\n"
                f"Evidencia disponible: {json.dumps(research.model_dump(), ensure_ascii=False)}\n"
                f"Feedback del Supervisor: {state.get('quality_feedback', '') or 'Ninguno.'}"
            )
            response = await self._model.ainvoke(
                [ANALYSIS_PROMPT, HumanMessage(content=instruction)]
            )
            if not response.tool_calls:
                raise ValueError("El Analista no generó la llamada obligatoria a su herramienta.")

            tool_call = response.tool_calls[0]
            raw_result = await calculate_percentage_change.ainvoke(tool_call["args"])
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
