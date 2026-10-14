"""Punto de entrada: configuración, demo offline, traza y SQLite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from multi_agent_orchestrator import main as main_module
from multi_agent_orchestrator.config import Settings


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    # load_dotenv() busca .env subiendo desde main.py, no desde el cwd: se anula para que
    # la prueba nunca lea la clave real del usuario.
    monkeypatch.setattr(main_module, "load_dotenv", lambda: False)
    for name in ("OPENAI_API_KEY", "THREAD_ID", "MAX_SUPERVISOR_STEPS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "data" / "cp.sqlite"))
    monkeypatch.setenv("TRACE_OUTPUT_PATH", str(tmp_path / "traces" / "trace.json"))
    return tmp_path


def test_settings_require_api_key_for_openai_mode(isolated_env: Path) -> None:
    del isolated_env
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        Settings.from_environment()


def test_settings_reject_invalid_step_limit(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del isolated_env
    monkeypatch.setenv("MAX_SUPERVISOR_STEPS", "muchos")
    with pytest.raises(RuntimeError, match="max_supervisor_steps"):
        Settings.from_environment(require_api_key=False)


def test_api_key_is_not_exposed_in_settings_repr(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del isolated_env
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-value")
    settings = Settings.from_environment()

    assert "sk-test-secret-value" not in repr(settings)


@pytest.mark.parametrize("steps", [4, 8, 20])
def test_recursion_limit_leaves_room_for_every_supervisor_step(steps: int) -> None:
    # Con N decisiones del Supervisor hay como máximo 2N - 1 supersteps.
    assert Settings(max_supervisor_steps=steps).recursion_limit > 2 * steps - 1


def test_offline_demo_creates_database_and_trace(
    isolated_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main_module.run(["--offline"])

    trace = json.loads((isolated_env / "traces" / "trace.json").read_text(encoding="utf-8"))
    assert (isolated_env / "data" / "cp.sqlite").exists()
    assert trace["mode"] == "offline"
    assert [event["node"] for event in trace["events"]] == [
        "supervisor",
        "researcher",
        "supervisor",
        "analyst",
        "supervisor",
        "synthesizer",
        "supervisor",
    ]
    assert "messages" not in json.dumps(trace)
    assert "INFORME FINAL" in capsys.readouterr().out


def test_run_without_api_key_exits_with_configuration_error(isolated_env: Path) -> None:
    del isolated_env
    with pytest.raises(SystemExit) as exit_info:
        main_module.run([])

    assert exit_info.value.code == 1
