"""Ejecuta y transmite una demostración completa de delegación jerárquica."""

from __future__ import annotations

import argparse
import asyncio
from typing import Any, cast

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import ValidationError

from multi_agent_orchestrator.agents import AnalystAgent, ResearchAgent, SynthesisAgent
from multi_agent_orchestrator.config import Settings
from multi_agent_orchestrator.graph import TeamNodes, build_graph
from multi_agent_orchestrator.offline import OfflineDemoModel
from multi_agent_orchestrator.state import AgentState, FinalReport
from multi_agent_orchestrator.supervisor import SupervisorNode
from multi_agent_orchestrator.tracing import print_event, save_trace, update_to_event

DEMO_REQUEST = (
    "Investigá el crecimiento del mercado de energía solar entre 2023 y 2024, "
    "calculá la variación porcentual y redactá una recomendación ejecutiva basada en fuentes."
)


def build_team(model: Any) -> TeamNodes:
    """Un mismo modelo, cuatro roles con prompts, herramientas y parcelas distintas."""

    return TeamNodes(
        supervisor=SupervisorNode(model),
        researcher=ResearchAgent(model),
        analyst=AnalystAgent(model),
        synthesizer=SynthesisAgent(model),
    )


def initial_state(request: str, max_steps: int) -> AgentState:
    return {
        "messages": [HumanMessage(content=request)],
        "request": request,
        "research_output": None,
        "analysis_output": None,
        "final_report": None,
        "task_completed": False,
        "step_count": 0,
        "max_steps": max_steps,
        "contributions": [],
        "errors": [],
    }


async def run_demo(
    model: Any,
    settings: Settings,
    request: str = DEMO_REQUEST,
    *,
    mode: str = "openai",
) -> AgentState:
    """Transmite cada nodo, persiste checkpoints en SQLite y guarda la traza."""

    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    config: RunnableConfig = {
        "configurable": {"thread_id": settings.thread_id},
        "recursion_limit": settings.recursion_limit,
    }

    events: list[dict[str, Any]] = []
    async with AsyncSqliteSaver.from_conn_string(
        str(settings.checkpoint_db_path)
    ) as checkpointer:
        await checkpointer.setup()
        graph = build_graph(nodes=build_team(model), checkpointer=checkpointer)

        print(f"\n=== DELEGACIÓN MULTI-AGENTE ({mode}) ===")
        async for update in graph.astream(
            initial_state(request, settings.max_supervisor_steps),
            config=config,
            stream_mode="updates",
        ):
            for node, values in cast(dict[str, dict[str, Any]], update).items():
                event = update_to_event(len(events) + 1, node, values)
                events.append(event)
                print_event(event)

        snapshot = await graph.aget_state(config)

    save_trace(events, settings.trace_output_path, settings.thread_id, mode=mode)
    return cast(AgentState, snapshot.values)


def print_result(final_state: AgentState, settings: Settings) -> bool:
    """Muestra el informe si la tarea terminó; devuelve si se completó."""

    try:
        report = FinalReport.model_validate(final_state.get("final_report"))
    except ValidationError:
        report = None

    if final_state.get("task_completed") and report is not None:
        print("\n=== INFORME FINAL ===")
        print(report.title)
        print(report.executive_summary)
        print(f"Recomendación: {report.recommendation}")
        completed = True
    else:
        print("\n=== TAREA INCOMPLETA ===")
        for error in final_state.get("errors", [])[-3:]:
            print(f"- {error}")
        completed = False

    print(f"\nThread: {settings.thread_id}")
    print(f"Traza: {settings.trace_output_path}")
    print(f"Checkpoints: {settings.checkpoint_db_path}")
    return completed


async def main(offline: bool = False) -> bool:
    """Construye el equipo y ejecuta la demo con OpenAI o con el modelo offline."""

    load_dotenv()
    settings = Settings.from_environment(require_api_key=not offline)
    model: Any
    if offline:
        model = OfflineDemoModel()
    else:
        model = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
    final_state = await run_demo(model, settings, mode="offline" if offline else "openai")
    return print_result(final_state, settings)


def run(argv: list[str] | None = None) -> None:
    """Punto de entrada del comando instalado."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Usa un modelo determinista local: no requiere API key ni consume tokens.",
    )
    args = parser.parse_args(argv)
    try:
        completed = asyncio.run(main(offline=args.offline))
    except RuntimeError as exc:
        print(f"Error de configuración: {exc}")
        raise SystemExit(1) from exc
    if not completed:
        raise SystemExit(2)


if __name__ == "__main__":
    run()
