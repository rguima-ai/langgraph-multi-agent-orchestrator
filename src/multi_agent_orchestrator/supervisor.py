"""Supervisor central que delega, controla calidad y decide cuándo finalizar."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from multi_agent_orchestrator.errors import RECOVERABLE_LLM_ERRORS
from multi_agent_orchestrator.state import AgentState, Route, SupervisorDecision

SUPERVISOR_PROMPT = SystemMessage(
    content=(
        "Sos el Supervisor de un equipo multi-agente. Coordinás; no investigás, no calculás y no "
        "redactás el informe. Elegí exactamente un próximo paso: researcher, analyst, synthesizer "
        "o FINISH. Rúbrica: researcher debe aportar al menos dos métricas numéricas con períodos, "
        "unidades, source_id y confianza >= 0.6; analyst debe calcular una variación, declarar la "
        "fórmula y confianza >= 0.8; synthesizer debe usar evidencia y análisis, citar fuentes, "
        "recomendar y declarar limitaciones. Si un artefacto es insuficiente, reenviá el trabajo "
        "al especialista correspondiente con quality_feedback concreto. Solo elegí FINISH si el "
        "informe final cumple toda la rúbrica."
    )
)


class SupervisorNode:
    """Nodo de ruteo mediante salida estructurada y una condición de parada defensiva."""

    def __init__(self, model: Any) -> None:
        self._model = model.with_structured_output(SupervisorDecision)

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        current_step = state.get("step_count", 0)
        max_steps = state.get("max_steps", 8)
        if current_step >= max_steps:
            reason = f"Se alcanzó el límite defensivo de {max_steps} decisiones del Supervisor."
            return {
                "next_agent": "FINISH",
                "supervisor_reason": reason,
                "task_completed": bool(state.get("final_report")),
                "errors": [reason],
                "messages": [AIMessage(content=reason, name="supervisor")],
            }

        snapshot = {
            "request": state["request"],
            "research_output": state.get("research_output"),
            "analysis_output": state.get("analysis_output"),
            "final_report": state.get("final_report"),
            "recent_errors": state.get("errors", [])[-3:],
            "step": current_step + 1,
            "max_steps": max_steps,
        }
        try:
            decision = await self._model.ainvoke(
                [
                    SUPERVISOR_PROMPT,
                    HumanMessage(content=json.dumps(snapshot, ensure_ascii=False, indent=2)),
                ]
            )
            validated = SupervisorDecision.model_validate(decision)
        except (ValidationError, *RECOVERABLE_LLM_ERRORS) as exc:
            fallback_route: Route
            if not state.get("research_output"):
                fallback_route = "researcher"
            elif not state.get("analysis_output"):
                fallback_route = "analyst"
            elif not state.get("final_report"):
                fallback_route = "synthesizer"
            else:
                fallback_route = "FINISH"

            error = f"Supervisor: {type(exc).__name__}: {exc}"
            validated = SupervisorDecision(
                next_agent=fallback_route,
                reason="Se aplicó el ruteo defensivo porque falló la decisión del LLM.",
                quality_feedback="Reintentá la tarea respetando el contrato de salida.",
            )
            snapshot["recoverable_error"] = error

        if validated.next_agent == "FINISH" and not state.get("final_report"):
            validated = SupervisorDecision(
                next_agent="synthesizer",
                reason="No existe un informe final; debe intervenir Síntesis antes de terminar.",
                quality_feedback="Completá el informe usando la evidencia y el análisis validados.",
            )

        message = f"Próximo agente: {validated.next_agent}. Motivo: {validated.reason}"
        update: dict[str, Any] = {
            "next_agent": validated.next_agent,
            "supervisor_reason": validated.reason,
            "quality_feedback": validated.quality_feedback,
            "task_completed": validated.next_agent == "FINISH",
            "step_count": current_step + 1,
            "messages": [AIMessage(content=message, name="supervisor")],
            "contributions": [
                {
                    "agent": "supervisor",
                    "status": "decision",
                    "next_agent": validated.next_agent,
                    "summary": validated.reason,
                }
            ],
        }
        if "recoverable_error" in snapshot:
            update["errors"] = [snapshot["recoverable_error"]]
        return update
