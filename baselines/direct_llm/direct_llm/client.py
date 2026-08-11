"""OpenAI-compatible API client helpers for the adapted Direct LLM baseline.

Credentials and endpoint URLs are read only from explicitly named environment
variables.  The OpenAI SDK is imported lazily so local data preparation and
CLI help do not require network dependencies or credentials.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping


_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass(frozen=True)
class APISettings:
    model: str = "qwen-plus"
    api_key_env: str = "LLM_API_KEY"
    base_url_env: str | None = "LLM_BASE_URL"
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout_seconds: float = 120.0
    max_api_retries: int = 4
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0

    def validate(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("llm.model must be a non-empty string")
        _validate_environment_name(self.api_key_env, "llm.api_key_env")
        if self.base_url_env:
            _validate_environment_name(self.base_url_env, "llm.base_url_env")
        if not math.isfinite(self.temperature) or self.temperature < 0.0:
            raise ValueError("llm.temperature must be finite and non-negative")
        if self.max_tokens <= 0:
            raise ValueError("llm.max_tokens must be positive")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError("llm.timeout_seconds must be positive and finite")
        if self.max_api_retries < 0:
            raise ValueError("llm.max_api_retries cannot be negative")
        if (
            not math.isfinite(self.initial_backoff_seconds)
            or not math.isfinite(self.max_backoff_seconds)
            or self.initial_backoff_seconds < 0.0
            or self.max_backoff_seconds < 0.0
        ):
            raise ValueError("API backoff durations must be finite and non-negative")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("max_backoff_seconds cannot be below initial_backoff_seconds")


@dataclass(frozen=True)
class CompletionResult:
    text: str
    api_attempts: int


class DirectLLMAPIError(RuntimeError):
    """Credential-safe terminal API failure."""

    def __init__(self, message: str, *, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


def settings_from_mapping(config: Mapping[str, Any]) -> APISettings:
    """Construct validated settings from the public ``llm`` config object."""

    if not isinstance(config, Mapping):
        raise ValueError("llm configuration must be a JSON object")
    raw_base_env = config.get("base_url_env", "LLM_BASE_URL")
    base_url_env = None if raw_base_env in (None, "") else str(raw_base_env)
    settings = APISettings(
        model=str(config.get("model", "qwen-plus")),
        api_key_env=str(config.get("api_key_env", "LLM_API_KEY")),
        base_url_env=base_url_env,
        temperature=float(config.get("temperature", 0.3)),
        max_tokens=int(config.get("max_tokens", 4096)),
        timeout_seconds=float(config.get("timeout_seconds", 120.0)),
        max_api_retries=int(config.get("max_api_retries", 4)),
        initial_backoff_seconds=float(config.get("initial_backoff_seconds", 1.0)),
        max_backoff_seconds=float(config.get("max_backoff_seconds", 30.0)),
    )
    settings.validate()
    return settings


def create_openai_client(
    settings: APISettings,
    *,
    environ: Mapping[str, str] | None = None,
    client_factory: Callable[..., Any] | None = None,
) -> Any:
    """Create an OpenAI>=1 client from environment-sourced secrets only.

    An empty or unset base-URL environment variable intentionally selects the
    SDK's default endpoint.  The API key remains required.
    """

    settings.validate()
    environment = os.environ if environ is None else environ
    api_key = environment.get(settings.api_key_env, "").strip()
    if not api_key:
        raise DirectLLMAPIError(
            f"required API credential environment variable is unset: {settings.api_key_env}",
            attempts=0,
        )
    base_url = ""
    if settings.base_url_env:
        base_url = environment.get(settings.base_url_env, "").strip()
    if client_factory is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("install 'openai>=1' to run Direct LLM inference") from exc
        client_factory = OpenAI
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": settings.timeout_seconds,
        # Retries are implemented below so their timing and audit count are explicit.
        "max_retries": 0,
    }
    if base_url:
        kwargs["base_url"] = base_url
    try:
        return client_factory(**kwargs)
    except Exception as exc:
        raise DirectLLMAPIError(
            f"failed to initialize API client: {_safe_exception_summary(exc)}",
            attempts=0,
        ) from None


def complete_with_retries(
    client: Any,
    settings: APISettings,
    *,
    system_prompt: str,
    user_prompt: str,
    format_reminder: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> CompletionResult:
    """Request one completion with bounded exponential API retry backoff."""

    settings.validate()
    system = _required_text(system_prompt, "system_prompt")
    user = _required_text(user_prompt, "user_prompt")
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if format_reminder is not None:
        reminder = _required_text(format_reminder, "format_reminder")
        # Keep a conventional single-turn system/user layout for compatible
        # endpoints that reject consecutive messages with the same role.
        messages[1]["content"] = f"{user}\n\n{reminder}"

    total_attempts = settings.max_api_retries + 1
    for attempt in range(1, total_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=settings.model,
                messages=messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                timeout=settings.timeout_seconds,
            )
            text = _response_text(response)
            return CompletionResult(text=text, api_attempts=attempt)
        except Exception as exc:
            retryable = _is_retryable(exc)
            if attempt >= total_attempts or not retryable:
                raise DirectLLMAPIError(
                    f"API request failed after {attempt} attempt(s): "
                    f"{_safe_exception_summary(exc)}",
                    attempts=attempt,
                ) from None
            delay = min(
                settings.max_backoff_seconds,
                settings.initial_backoff_seconds * (2 ** (attempt - 1)),
            )
            sleep(delay)
    raise AssertionError("unreachable retry state")


def _response_text(response: Any) -> str:
    try:
        choices = response.choices
        content = choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("API response lacks choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("API returned an empty text response")
    return content.strip()


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        # Connection/timeouts and malformed transient responses do not always
        # expose a status code. Boundaries remain capped by max_api_retries.
        return True
    try:
        numeric = int(status)
    except (TypeError, ValueError):
        return True
    return numeric in {408, 409, 425, 429} or numeric >= 500


def _safe_exception_summary(exc: Exception) -> str:
    """Summarize failures without including server text, URLs, or credentials."""

    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    try:
        numeric = int(status) if status is not None else None
    except (TypeError, ValueError):
        numeric = None
    return f"{name} (HTTP {numeric})" if numeric is not None else name


def _validate_environment_name(value: Any, field: str) -> None:
    if not isinstance(value, str) or _ENVIRONMENT_NAME.fullmatch(value) is None:
        raise ValueError(f"{field} must be a valid environment-variable name")


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()
