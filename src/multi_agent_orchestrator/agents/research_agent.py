"""Especialista que recupera evidencia sin modificar otras parcelas del estado."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from multi_agent_orchestrator.errors import RECOVERABLE_LLM_ERRORS
from multi_agent_orchestrator.state import AgentState, ResearchArtifact
from multi_agent_orchestrator.tools import (
    RESEARCH_TOOLS,
    raise_if_tool_error,
    search_market_data,
)

RESEARCH_PROMPT = SystemMessage(
    content=(
        "Sos el agente Investigador. Tu única responsabilidad es buscar evidencia verificable. "
        "Usá search_market_data una vez con una consulta que incluya el tema y los períodos. "
        "No calcules porcentajes, no redactes la conclusión final y no inventes fuentes."
    )
)


class ResearchAgent:
    """Nodo async con acceso exclusivo a la herramienta de investigación."""

    def __init__(self, model: Any) -> None:
        self._model = model.bind_tools(
            RESEARCH_TOOLS,
            tool_choice=search_market_data.name,
        )

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        try:
            # Contexto mínimo: el pedido y el feedback; no ve análisis ni informe.
            payload = {
                "request": state["request"],
                "supervisor_feedback": state.get("quality_feedback", ""),
            }
            response = await self._model.ainvoke(
                [RESEARCH_PROMPT, HumanMessage(content=json.dumps(payload, ensure_ascii=False))]
            )
            if not response.tool_calls:
                raise ValueError(
                    "El Investigador no generó la llamada obligatoria a su herramienta."
                )

            tool_call = response.tool_calls[0]
            raw_result = await search_market_data.ainvoke(tool_call["args"])
            raise_if_tool_error(raw_result)
            artifact = ResearchArtifact.model_validate_json(raw_result)
            summary = (
                f"Investigación validada: {len(artifact.findings)} hallazgos, "
                f"confianza {artifact.confidence:.0%}."
            )
            return {
                "research_output": artifact.model_dump(mode="json"),
                "messages": [AIMessage(content=summary, name="researcher")],
                "contributions": [
                    {
                        "agent": "researcher",
                        "status": "success",
                        "summary": summary,
                        "source_ids": [item.source_id for item in artifact.findings],
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
            error = f"Investigador: {type(exc).__name__}: {exc}"
            return {
                "messages": [AIMessage(content=error, name="researcher")],
                "errors": [error],
                "contributions": [
                    {"agent": "researcher", "status": "error", "summary": error}
                ],
            }
