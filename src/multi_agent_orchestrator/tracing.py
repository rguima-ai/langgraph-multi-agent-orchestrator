"""Registro auditable de la delegación sin guardar razonamiento privado."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage


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
    """Conserva decisiones, artefactos y errores observables."""

    return {
        "position": position,
        "node": node,
        "next_agent": update.get("next_agent"),
        "supervisor_reason": update.get("supervisor_reason"),
        "research_output": _json_safe(update.get("research_output")),
        "analysis_output": _json_safe(update.get("analysis_output")),
        "final_report": _json_safe(update.get("final_report")),
        "contributions": _json_safe(update.get("contributions", [])),
        "errors": _json_safe(update.get("errors", [])),
    }


def save_trace(
    events: list[dict[str, Any]],
    output_path: Path,
    thread_id: str,
) -> None:
    """Escribe la traza en UTF-8 sin incluir prompts internos ni secretos."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "thread_id": thread_id,
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
        print(f"[SUPERVISOR] -> {event['next_agent']}: {event['supervisor_reason']}")
    elif event["errors"]:
        print(f"[{node.upper()}] ERROR: {event['errors'][-1]}")
    else:
        print(f"[{node.upper()}] trabajo completado")
