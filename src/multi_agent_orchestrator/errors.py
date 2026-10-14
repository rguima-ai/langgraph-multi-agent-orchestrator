"""Excepciones recuperables que no deben derribar el workflow."""

from langchain_openai.chat_models.base import OpenAIRefusalError
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    LengthFinishReasonError,
    RateLimitError,
)

# Fallos transitorios o de formato: el nodo los registra en `errors` y el grafo sigue.
# Errores de configuración (por ejemplo, AuthenticationError) no se capturan a propósito:
# reintentar con una API key inválida no tiene sentido.
RECOVERABLE_LLM_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    LengthFinishReasonError,
    OpenAIRefusalError,
    TimeoutError,
    ConnectionError,
)
