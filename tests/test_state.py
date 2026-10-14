"""Contratos Pydantic: un artefacto inválido nunca entra al estado como válido."""

from typing import Any

import pytest
from conftest import analysis_output, research_output
from pydantic import ValidationError

from multi_agent_orchestrator.state import (
    ROUTES,
    AnalysisArtifact,
    ResearchArtifact,
    SupervisorDecision,
)


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_research_artifact_rejects_confidence_out_of_range(confidence: float) -> None:
    # Artefacto válido salvo la confianza: así la prueba falla por el motivo correcto.
    with pytest.raises(ValidationError, match="confidence"):
        ResearchArtifact.model_validate(research_output(confidence=confidence))


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_research_artifact_accepts_confidence_limits(confidence: float) -> None:
    assert ResearchArtifact.model_validate(research_output(confidence)).confidence == confidence


def test_analysis_artifact_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValidationError, match="confidence"):
        AnalysisArtifact.model_validate({**analysis_output(), "confidence": 2})


def test_analysis_artifact_rejects_result_inconsistent_with_formula() -> None:
    with pytest.raises(ValidationError, match="no coincide"):
        AnalysisArtifact.model_validate({**analysis_output(), "result": 25.0})


def test_research_artifact_rejects_empty_and_duplicated_findings() -> None:
    data: dict[str, Any] = research_output()
    with pytest.raises(ValidationError):
        ResearchArtifact.model_validate({**data, "findings": []})
    with pytest.raises(ValidationError, match="duplicados"):
        ResearchArtifact.model_validate({**data, "findings": data["findings"][:1] * 2})


def test_supervisor_decision_only_accepts_known_routes() -> None:
    assert ROUTES == ("researcher", "analyst", "synthesizer", "FINISH")
    with pytest.raises(ValidationError):
        SupervisorDecision.model_validate(
            {"next_agent": "writer", "reason": "Ruta inexistente.", "quality_feedback": ""}
        )
