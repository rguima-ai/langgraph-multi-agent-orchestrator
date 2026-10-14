"""Especialista que sintetiza los artefactos sin realizar nuevas búsquedas."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from multi_agent_orchestrator.errors import RECOVERABLE_LLM_ERRORS
from multi_agent_orchestrator.state import (
    AgentState,
    AnalysisArtifact,
    FinalReport,
    ResearchArtifact,
)

SYNTHESIS_PROMPT = SystemMessage(
    content=(
        "Sos el agente de Síntesis. Redactá un informe ejecutivo usando únicamente la evidencia "
        "y el análisis suministrados. Citá los source_id, incluí la cifra calculada, proponé una "
        "recomendación y declará al menos una limitación. No hagas nuevas búsquedas."
    )
)


def _cited_source_ids(research: ResearchArtifact, report: FinalReport) -> list[str]:
    """Exige que el informe cite al menos una fuente real de la investigación."""

    text = " ".join([*report.evidence, report.executive_summary, report.recommendation])
    cited = sorted({f.source_id for f in research.findings if f.source_id in text})
    if not cited:
        raise ValueError("El informe no cita ningún source_id de la investigación.")
    return cited


class SynthesisAgent:
    """Nodo async que produce un `FinalReport` validado con Pydantic."""

    def __init__(self, model: Any) -> None:
        self._model = model.with_structured_output(FinalReport)

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        try:
            research = ResearchArtifact.model_validate(state.get("research_output"))
            analysis = AnalysisArtifact.model_validate(state.get("analysis_output"))
            payload = {
                "request": state["request"],
                "research": research.model_dump(mode="json"),
                "analysis": analysis.model_dump(mode="json"),
                "supervisor_feedback": state.get("quality_feedback", ""),
            }
            report = await self._model.ainvoke(
                [
                    SYNTHESIS_PROMPT,
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
                ]
            )
            validated = FinalReport.model_validate(report)
            cited = _cited_source_ids(research, validated)
            summary = f"Informe final validado: {validated.title}"
            return {
                "final_report": validated.model_dump(mode="json"),
                "messages": [
                    AIMessage(content=validated.executive_summary, name="synthesizer")
                ],
                "contributions": [
                    {
                        "agent": "synthesizer",
                        "status": "success",
                        "summary": summary,
                        "source_ids": cited,
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
            error = f"Síntesis: {type(exc).__name__}: {exc}"
            return {
                "messages": [AIMessage(content=error, name="synthesizer")],
                "errors": [error],
                "contributions": [
                    {"agent": "synthesizer", "status": "error", "summary": error}
                ],
            }
