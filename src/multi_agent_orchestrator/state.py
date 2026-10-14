"""Contratos del estado compartido y de los artefactos de cada especialista."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, NotRequired

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

Route = Literal["researcher", "analyst", "synthesizer", "FINISH"]


class ResearchFinding(BaseModel):
    """Dato individual recuperado por el Investigador."""

    label: str = Field(min_length=3)
    value: float
    unit: str = Field(min_length=1)
    period: str = Field(min_length=2)
    source_id: str = Field(min_length=3)
    note: str = ""


class ResearchArtifact(BaseModel):
    """Salida validada que pertenece exclusivamente al Investigador."""

    query: str = Field(min_length=3)
    findings: list[ResearchFinding] = Field(min_length=1)
    summary: str = Field(min_length=10)
    confidence: float = Field(ge=0.0, le=1.0)


class AnalysisArtifact(BaseModel):
    """Salida validada que pertenece exclusivamente al Analista."""

    metric: str = Field(min_length=3)
    start_value: float
    end_value: float
    result: float
    formula: str = Field(min_length=3)
    interpretation: str = Field(min_length=10)
    confidence: float = Field(ge=0.0, le=1.0)


class FinalReport(BaseModel):
    """Informe final producido después de investigación y análisis."""

    title: str = Field(min_length=5)
    executive_summary: str = Field(min_length=30)
    evidence: list[str] = Field(min_length=2)
    recommendation: str = Field(min_length=20)
    limitations: list[str] = Field(min_length=1)


class SupervisorDecision(BaseModel):
    """Contrato estructurado para el ruteo dinámico del Supervisor."""

    next_agent: Route
    reason: str = Field(min_length=10)
    quality_feedback: str = ""


class AgentState(MessagesState, total=False):
    """Estado compartido con parcelas de escritura claramente separadas."""

    request: str
    research_output: NotRequired[dict[str, Any] | None]
    analysis_output: NotRequired[dict[str, Any] | None]
    final_report: NotRequired[dict[str, Any] | None]
    next_agent: NotRequired[Route]
    supervisor_reason: NotRequired[str]
    quality_feedback: NotRequired[str]
    task_completed: NotRequired[bool]
    step_count: NotRequired[int]
    max_steps: NotRequired[int]
    contributions: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
