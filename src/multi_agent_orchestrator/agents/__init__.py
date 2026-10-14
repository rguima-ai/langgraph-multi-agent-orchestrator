"""Agentes especialistas con responsabilidades y herramientas acotadas."""

from multi_agent_orchestrator.agents.analyst_agent import AnalystAgent
from multi_agent_orchestrator.agents.research_agent import ResearchAgent
from multi_agent_orchestrator.agents.synthesis_agent import SynthesisAgent

__all__ = ["AnalystAgent", "ResearchAgent", "SynthesisAgent"]
