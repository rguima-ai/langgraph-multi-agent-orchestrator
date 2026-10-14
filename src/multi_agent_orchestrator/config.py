"""Configuración validada desde variables de entorno."""

import os
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, ValidationError


def _default_thread_id() -> str:
    """Un hilo nuevo por ejecución evita mezclar contribuciones de corridas anteriores."""

    return f"multi-agent-demo-{datetime.now(UTC):%Y%m%dT%H%M%S}"


class Settings(BaseModel):
    """Valores configurables sin incluir secretos en el código."""

    # SecretStr evita que la clave aparezca al imprimir o loguear la configuración.
    openai_api_key: SecretStr | None = None
    openai_model: str = Field(default="gpt-4.1-mini", min_length=1)
    checkpoint_db_path: Path = Path("data/checkpoints.sqlite")
    trace_output_path: Path = Path("traces/latest_trace.json")
    thread_id: str = Field(default_factory=_default_thread_id, min_length=1)
    max_supervisor_steps: int = Field(default=8, ge=4, le=20)

    @property
    def recursion_limit(self) -> int:
        """Cada decisión del Supervisor puede ir seguida de un especialista: 2 supersteps."""

        return self.max_supervisor_steps * 2 + 2

    @classmethod
    def from_environment(cls, *, require_api_key: bool = True) -> "Settings":
        """Carga y valida el entorno sin imprimir la API key."""

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if require_api_key and not api_key:
            raise RuntimeError(
                "Falta OPENAI_API_KEY. Copiá .env.example como .env y agregá tu clave, "
                "o ejecutá la demo sin red con --offline."
            )

        values: dict[str, object] = {
            "openai_api_key": api_key or None,
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            "checkpoint_db_path": os.getenv("CHECKPOINT_DB_PATH", "data/checkpoints.sqlite"),
            "trace_output_path": os.getenv("TRACE_OUTPUT_PATH", "traces/latest_trace.json"),
            "max_supervisor_steps": os.getenv("MAX_SUPERVISOR_STEPS", "8"),
        }
        if thread_id := os.getenv("THREAD_ID", "").strip():
            values["thread_id"] = thread_id

        try:
            return cls.model_validate(values)
        except ValidationError as exc:
            fields = ", ".join(str(error["loc"][0]) for error in exc.errors())
            raise RuntimeError(f"Configuración inválida en: {fields}.") from exc
