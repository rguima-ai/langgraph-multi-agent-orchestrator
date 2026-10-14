"""Construcción del grafo jerárquico Supervisor -> Especialistas -> Supervisor."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from multi_agent_orchestrator.state import ROUTES, AgentState, Route

Node = Callable[[AgentState], Awaitable[Any]]


@dataclass(frozen=True)
class TeamNodes:
    """Dependencias inyectables: producción usa LLMs y tests usan nodos simulados."""

    supervisor: Node
    researcher: Node
    analyst: Node
    synthesizer: Node


def route_supervisor(state: AgentState) -> Route:
    """Lee la decisión estructurada; no contiene reglas sobre el texto del usuario.

    Las dos guardas protegen frente a un Supervisor defectuoso (por ejemplo, un doble de
    prueba): una ruta desconocida o un contador agotado terminan el grafo en vez de fallar.
    """

    if state.get("step_count", 0) >= state.get("max_steps", 8):
        return "FINISH"
    next_agent = state.get("next_agent", "FINISH")
    if next_agent not in ROUTES:
        return "FINISH"
    return next_agent


def build_graph(
    nodes: TeamNodes,
    checkpointer: Any | None = None,
) -> CompiledStateGraph[AgentState, None, Any, Any]:
    """Compila una topología jerárquica con retorno al Supervisor."""

    builder = StateGraph(AgentState)
    builder.add_node("supervisor", RunnableLambda(nodes.supervisor))
    builder.add_node("researcher", RunnableLambda(nodes.researcher))
    builder.add_node("analyst", RunnableLambda(nodes.analyst))
    builder.add_node("synthesizer", RunnableLambda(nodes.synthesizer))

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "synthesizer": "synthesizer",
            "FINISH": END,
        },
    )
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("analyst", "supervisor")
    builder.add_edge("synthesizer", "supervisor")

    return builder.compile(checkpointer=checkpointer)
