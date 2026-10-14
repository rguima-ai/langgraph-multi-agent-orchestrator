"""Excepciones recuperables que no deben derribar el workflow."""

from openai import APIConnectionError, APITimeoutError, RateLimitError

RECOVERABLE_LLM_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    TimeoutError,
    ConnectionError,
)
