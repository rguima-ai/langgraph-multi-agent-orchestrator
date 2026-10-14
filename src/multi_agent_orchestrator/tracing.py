"""Registro auditable de la delegación sin guardar razonamiento privado."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage

# Campos estructurados que se copian a la traza; `messages` queda afuera a propósito.
TRACED_FIELDS = (
    "step_count",
    "next_agent",
    "supervisor_reason",
    "quality_feedback",
    "task_completed",
    "research_output",
    "analysis_output",
    "final_report",
    "contributions",
    "errors",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        return {
            "type": value.type,
            "name": value.name,
            "content": value.content,
        }
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def update_to_event(position: int, node: str, update: dict[str, Any]) -> dict[str, Any]:
    """Conserva decisiones, artefactos y errores observables.

    Por diseño no copia `messages` ni prompts: solo campos estructurados del estado. El
    `supervisor_reason` es la justificación breve que el contrato pide de forma explícita,
    no el razonamiento interno del modelo.
    """

    event: dict[str, Any] = {"position": position, "node": node}
    for field in TRACED_FIELDS:
        if field in update:
            event[field] = _json_safe(update[field])
    event.setdefault("contributions", [])
    event.setdefault("errors", [])
    return event


def save_trace(
    events: list[dict[str, Any]],
    output_path: Path,
    thread_id: str,
    *,
    mode: str = "openai",
) -> None:
    """Escribe la traza en UTF-8 sin incluir prompts internos ni secretos."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "thread_id": thread_id,
        "mode": mode,
        "generated_at": datetime.now(UTC).isoformat(),
        "event_count": len(events),
        "events": events,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_event(event: dict[str, Any]) -> None:
    """Muestra una traza breve durante el streaming."""

    node = event["node"]
    if node == "supervisor":
        print(f"[SUPERVISOR] -> {event.get('next_agent')}: {event.get('supervisor_reason')}")
    elif event["errors"]:
        print(f"[{node.upper()}] ERROR: {event['errors'][-1]}")
    else:
        print(f"[{node.upper()}] trabajo completado")
