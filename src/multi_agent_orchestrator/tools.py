"""Herramientas acotadas que utilizan los agentes especialistas."""

from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from multi_agent_orchestrator.state import AnalysisArtifact, ResearchArtifact, ResearchFinding


class SearchInput(BaseModel):
    """Entrada validada para la búsqueda simulada."""

    query: str = Field(
        min_length=3,
        max_length=220,
        description="Tema y períodos concretos que deben investigarse.",
    )
    limit: int = Field(default=4, ge=1, le=6)


class GrowthInput(BaseModel):
    """Entrada validada para el cálculo de variación porcentual."""

    start_value: float = Field(description="Valor inicial del período base.")
    end_value: float = Field(description="Valor final del período comparado.")
    start_period: str = Field(min_length=2)
    end_period: str = Field(min_length=2)


class MockKnowledgeBase:
    """Fuente local reproducible que reemplaza una API de búsqueda en esta entrega."""

    def __init__(self) -> None:
        self._documents: list[dict[str, Any]] = [
            {
                "label": "Tamaño del mercado solar",
                "value": 120.0,
                "unit": "millones de USD",
                "period": "2023",
                "source_id": "market-report-2023",
                "note": "Estimación consolidada del mercado simulado para 2023.",
                "tags": "energia solar mercado crecimiento 2023 tamaño ventas",
            },
            {
                "label": "Tamaño del mercado solar",
                "value": 144.0,
                "unit": "millones de USD",
                "period": "2024",
                "source_id": "market-report-2024",
                "note": "Estimación consolidada del mercado simulado para 2024.",
                "tags": "energia solar mercado crecimiento 2024 tamaño ventas",
            },
            {
                "label": "Reducción del costo de instalación",
                "value": 8.0,
                "unit": "porcentaje",
                "period": "2024",
                "source_id": "solar-cost-index-2024",
                "note": "Variación interanual simulada del costo promedio de instalación.",
                "tags": "energia solar costo instalacion 2024 reduccion porcentaje",
            },
            {
                "label": "Satisfacción de clientes",
                "value": 4.6,
                "unit": "puntos sobre 5",
                "period": "2024",
                "source_id": "customer-survey-2024",
                "note": "Resultado de una encuesta simulada.",
                "tags": "clientes satisfaccion encuesta 2024",
            },
        ]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stop_words = {"a", "de", "del", "el", "en", "la", "las", "los", "por", "y"}
        normalized = unicodedata.normalize("NFKD", text.lower())
        without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
        return {
            token
            for token in re.findall(r"[a-z0-9]+", without_accents)
            if token not in stop_words
        }

    async def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Simula I/O y devuelve documentos ordenados por coincidencia léxica."""

        await asyncio.sleep(0)
        if query == "__simulate_timeout__":
            raise TimeoutError("La fuente de investigación no respondió a tiempo.")

        query_tokens = self._tokens(query)
        scored: list[tuple[int, dict[str, Any]]] = []
        for document in self._documents:
            searchable = f"{document['label']} {document['note']} {document['tags']}"
            score = len(query_tokens & self._tokens(searchable))
            if score > 0:
                scored.append((score, document))

        scored.sort(key=lambda item: (item[0], item[1]["period"]), reverse=True)
        return [document for _, document in scored[:limit]]


knowledge_base = MockKnowledgeBase()


@tool(args_schema=SearchInput)
async def search_market_data(query: str, limit: int = 4) -> str:
    """Busca métricas verificables en la base interna simulada.

    Usá esta herramienta para recuperar valores, períodos, unidades y fuentes antes de analizar
    una tendencia. Incluí el tema y los años en `query`. No inventes datos que no aparezcan en
    los resultados.
    """

    try:
        documents = await knowledge_base.search(query=query, limit=limit)
        if not documents:
            return json.dumps(
                {
                    "status": "not_found",
                    "query": query,
                    "message": "No hubo coincidencias; reformulá la búsqueda.",
                },
                ensure_ascii=False,
            )

        findings = [
            ResearchFinding.model_validate(
                {key: value for key, value in document.items() if key != "tags"}
            )
            for document in documents
        ]
        artifact = ResearchArtifact(
            query=query,
            findings=findings,
            summary=f"Se recuperaron {len(findings)} métricas con fuente identificada.",
            confidence=0.9 if len(findings) >= 2 else 0.6,
        )
        return artifact.model_dump_json()
    except (TimeoutError, ConnectionError) as exc:
        return json.dumps(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": "La búsqueda falló de forma controlada; el Supervisor puede reintentar.",
            },
            ensure_ascii=False,
        )


@tool(args_schema=GrowthInput)
async def calculate_percentage_change(
    start_value: float,
    end_value: float,
    start_period: str,
    end_period: str,
) -> str:
    """Calcula y explica la variación porcentual entre dos métricas comparables.

    Usá esta herramienta únicamente cuando ambos valores tengan la misma unidad y representen
    períodos diferentes. El cálculo es `(final - inicial) / inicial * 100`.
    """

    await asyncio.sleep(0)
    if start_value == 0:
        return json.dumps(
            {
                "status": "error",
                "error_type": "ZeroDivisionError",
                "message": "El valor inicial no puede ser cero.",
            },
            ensure_ascii=False,
        )

    result = ((end_value - start_value) / start_value) * 100
    artifact = AnalysisArtifact(
        metric=f"Variación porcentual {start_period}-{end_period}",
        start_value=start_value,
        end_value=end_value,
        result=round(result, 2),
        formula="((valor_final - valor_inicial) / valor_inicial) * 100",
        interpretation=(
            f"El indicador cambió {result:+.2f}% entre {start_period} y {end_period}."
        ),
        confidence=1.0,
    )
    return artifact.model_dump_json()


RESEARCH_TOOLS = [search_market_data]
ANALYSIS_TOOLS = [calculate_percentage_change]
