"""Ensamblado e integración con el `ChatOpenAI` real, sin red ni consumo de tokens.

Los dobles simples codifican nuestras suposiciones sobre LangChain. Acá se usa la clase real
y solo se reemplaza `_agenerate` (el punto donde saldría la petición HTTP), para comprobar que
`bind_tools(tool_choice=...)` y `with_structured_output()` funcionan con la versión instalada.
"""

from __future__ import annotations

from typing import Any

import pytest
from conftest import make_state
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from multi_agent_orchestrator import offline
from multi_agent_orchestrator.graph import build_graph
from multi_agent_orchestrator.main import build_team
from multi_agent_orchestrator.state import FinalReport, SupervisorDecision

FAKE_KEY = SecretStr("sk-test-not-a-real-key")


def make_model() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=FAKE_KEY, max_retries=0)


def test_assembly_compiles_graph_and_mermaid_without_network() -> None:
    graph = build_graph(build_team(make_model()))
    mermaid = graph.get_graph().draw_mermaid()

    assert set(graph.get_graph().nodes) == {
        "__start__",
        "supervisor",
        "researcher",
        "analyst",
        "synthesizer",
        "__end__",
    }
    assert "__start__ --> supervisor" in mermaid
    for specialist in ("researcher", "analyst", "synthesizer"):
        assert f"supervisor -.-> {specialist}" in mermaid
        assert f"{specialist} --> supervisor" in mermaid
    assert "supervisor -. &nbsp;FINISH&nbsp; .-> __end__" in mermaid


@pytest.mark.asyncio
async def test_full_flow_through_real_chat_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[str] = []

    async def fake_agenerate(
        self: ChatOpenAI, messages: list[BaseMessage], stop: Any = None, **kwargs: Any
    ) -> ChatResult:
        payload = offline._payload(messages)
        schema = kwargs.get("response_format")
        if schema is SupervisorDecision or schema is FinalReport:
            requests.append(f"structured:{schema.__name__}")
            parsed = (
                offline.supervisor_decision(payload)
                if schema is SupervisorDecision
                else offline.final_report(payload)
            )
            message = AIMessage(
                content=parsed.model_dump_json(), additional_kwargs={"parsed": parsed}
            )
        else:
            tool_name = kwargs["tool_choice"]["function"]["name"]
            assert [t["function"]["name"] for t in kwargs["tools"]] == [tool_name]
            requests.append(f"tool:{tool_name}")
            args = (
                offline.research_tool_args(payload)
                if tool_name == "search_market_data"
                else offline.analysis_tool_args(payload)
            )
            message = AIMessage(
                content="", tool_calls=[{"name": tool_name, "args": args, "id": "call_1"}]
            )
        return ChatResult(generations=[ChatGeneration(message=message)])

    monkeypatch.setattr(ChatOpenAI, "_agenerate", fake_agenerate)
    graph = build_graph(build_team(make_model()))

    result = await graph.ainvoke(make_state(), config={"recursion_limit": 18})

    assert requests == [
        "structured:SupervisorDecision",
        "tool:search_market_data",
        "structured:SupervisorDecision",
        "tool:calculate_percentage_change",
        "structured:SupervisorDecision",
        "structured:FinalReport",
        "structured:SupervisorDecision",
    ]
    assert result["task_completed"] is True
    assert result["errors"] == []
    assert result["analysis_output"]["result"] == 20.0
