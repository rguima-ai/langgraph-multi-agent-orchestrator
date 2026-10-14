"""Ejecuta y transmite una demostración completa de delegación jerárquica."""

from __future__ import annotations

import asyncio
from typing import Any, cast

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import SecretStr

from multi_agent_orchestrator.agents import AnalystAgent, ResearchAgent, SynthesisAgent
from multi_agent_orchestrator.config import Settings
from multi_agent_orchestrator.graph import TeamNodes, build_graph
from multi_agent_orchestrator.state import AgentState, FinalReport
from multi_agent_orchestrator.supervisor import SupervisorNode
from multi_agent_orchestrator.tracing import print_event, save_trace, update_to_event

DEMO_REQUEST = (
    "Investigá el crecimiento del mercado de energía solar entre 2023 y 2024, "
    "calculá la variación porcentual y redactá una recomendación ejecutiva basada en fuentes."
)


async def main() -> None:
    """Construye el equipo, transmite nodos y guarda checkpoints y traza."""

    load_dotenv()
    settings = Settings.from_environment()
    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)

    model = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=SecretStr(settings.openai_api_key),
    )
    nodes = TeamNodes(
        supervisor=SupervisorNode(model),
        researcher=ResearchAgent(model),
        analyst=AnalystAgent(model),
        synthesizer=SynthesisAgent(model),
    )
    initial_state: AgentState = {
        "messages": [HumanMessage(content=DEMO_REQUEST)],
        "request": DEMO_REQUEST,
        "research_output": None,
        "analysis_output": None,
        "final_report": None,
        "task_completed": False,
        "step_count": 0,
        "max_steps": settings.max_supervisor_steps,
        "contributions": [],
        "errors": [],
    }
    config: RunnableConfig = {
        "configurable": {"thread_id": settings.thread_id},
        "recursion_limit": 24,
    }

    events: list[dict[str, Any]] = []
    async with AsyncSqliteSaver.from_conn_string(
        str(settings.checkpoint_db_path)
    ) as checkpointer:
        await checkpointer.setup()
        graph = build_graph(nodes=nodes, checkpointer=checkpointer)

        print("\n=== DELEGACIÓN MULTI-AGENTE ===")
        async for update in graph.astream(initial_state, config=config, stream_mode="updates"):
            for node, values in cast(dict[str, dict[str, Any]], update).items():
                event = update_to_event(len(events) + 1, node, values)
                events.append(event)
                print_event(event)

        snapshot = await graph.aget_state(config)

    final_state = cast(AgentState, snapshot.values)
    report = FinalReport.model_validate(final_state.get("final_report"))
    save_trace(events, settings.trace_output_path, settings.thread_id)

    print("\n=== INFORME FINAL ===")
    print(report.title)
    print(report.executive_summary)
    print(f"Recomendación: {report.recommendation}")
    print(f"\nTraza: {settings.trace_output_path}")
    print(f"Checkpoints: {settings.checkpoint_db_path}")


def run() -> None:
    """Punto de entrada del comando instalado."""

    try:
        asyncio.run(main())
    except RuntimeError as exc:
        print(f"Error de configuración: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    run()
