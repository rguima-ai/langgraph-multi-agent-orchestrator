"""Contratos del estado compartido y de los artefactos de cada especialista."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, NotRequired, Self, get_args

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field, model_validator

Route = Literal["researcher", "analyst", "synthesizer", "FINISH"]
ROUTES: tuple[Route, ...] = get_args(Route)


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

    @model_validator(mode="after")
    def _unique_sources_per_period(self) -> Self:
        keys = [(item.source_id, item.period, item.label) for item in self.findings]
        if len(keys) != len(set(keys)):
            raise ValueError("La investigación contiene hallazgos duplicados.")
        return self


class AnalysisArtifact(BaseModel):
    """Salida validada que pertenece exclusivamente al Analista."""

    metric: str = Field(min_length=3)
    start_value: float
    end_value: float
    result: float
    formula: str = Field(min_length=3)
    interpretation: str = Field(min_length=10)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _result_matches_formula(self) -> Self:
        """Impide que entre al estado un porcentaje que no surge de los valores declarados."""

        if self.start_value == 0:
            raise ValueError("El valor inicial no puede ser cero.")
        expected = (self.end_value - self.start_value) / self.start_value * 100
        if abs(expected - self.result) > 0.01:
            raise ValueError(
                f"El resultado {self.result} no coincide con la fórmula ({expected:.2f})."
            )
        return self


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
    # Sin valor por defecto: el modo estricto de OpenAI exige que todos los campos sean
    # obligatorios. Si no hay correcciones, el modelo devuelve una cadena vacía.
    quality_feedback: str = Field(
        description="Corrección concreta para el próximo especialista, o cadena vacía."
    )


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
