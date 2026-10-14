"""Configuración validada desde variables de entorno."""

from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Valores configurables sin incluir secretos en el código."""

    openai_api_key: str = Field(min_length=1)
    openai_model: str = "gpt-4.1-mini"
    checkpoint_db_path: Path = Path("data/checkpoints.sqlite")
    trace_output_path: Path = Path("traces/latest_trace.json")
    thread_id: str = Field(default="multi-agent-demo-001", min_length=1)
    max_supervisor_steps: int = Field(default=8, ge=4, le=20)

    @classmethod
    def from_environment(cls) -> "Settings":
        """Carga y valida el entorno sin imprimir la API key."""

        import os

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "Falta OPENAI_API_KEY. Copiá .env.example como .env y agregá tu clave."
            )

        return cls(
            openai_api_key=api_key,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            checkpoint_db_path=Path(
                os.getenv("CHECKPOINT_DB_PATH", "data/checkpoints.sqlite")
            ),
            trace_output_path=Path(
                os.getenv("TRACE_OUTPUT_PATH", "traces/latest_trace.json")
            ),
            thread_id=os.getenv("THREAD_ID", "multi-agent-demo-001"),
            max_supervisor_steps=int(os.getenv("MAX_SUPERVISOR_STEPS", "8")),
        )
