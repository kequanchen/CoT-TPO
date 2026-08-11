from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from direct_llm.client import (  # noqa: E402
    APISettings,
    DirectLLMAPIError,
    complete_with_retries,
    create_openai_client,
)


def response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


class HTTPError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.endpoint = FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.endpoint)


class ClientTests(unittest.TestCase):
    def test_nonfinite_numeric_settings_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            APISettings(temperature=float("nan")).validate()
        with self.assertRaises(ValueError):
            APISettings(timeout_seconds=float("inf")).validate()
        with self.assertRaises(ValueError):
            APISettings(initial_backoff_seconds=float("nan")).validate()

    def test_client_reads_key_and_optional_url_only_from_environment(self) -> None:
        captured = {}

        def factory(**kwargs):
            captured.update(kwargs)
            return object()

        settings = APISettings(api_key_env="SAFE_KEY", base_url_env="SAFE_URL")
        create_openai_client(
            settings,
            environ={"SAFE_KEY": "secret-value", "SAFE_URL": "https://example.invalid/v1"},
            client_factory=factory,
        )
        self.assertEqual(captured["api_key"], "secret-value")
        self.assertEqual(captured["base_url"], "https://example.invalid/v1")
        self.assertEqual(captured["max_retries"], 0)

        captured.clear()
        create_openai_client(
            settings,
            environ={"SAFE_KEY": "secret-value", "SAFE_URL": ""},
            client_factory=factory,
        )
        self.assertNotIn("base_url", captured)

    def test_missing_key_error_names_variable_but_never_contains_secret(self) -> None:
        with self.assertRaises(DirectLLMAPIError) as caught:
            create_openai_client(
                APISettings(api_key_env="REQUIRED_KEY"),
                environ={},
                client_factory=lambda **kwargs: object(),
            )
        self.assertIn("REQUIRED_KEY", str(caught.exception))

    def test_client_initialization_error_does_not_echo_key_or_url(self) -> None:
        secret = "DO_NOT_LOG_CREDENTIAL_SENTINEL"
        url = "https://private-endpoint.example.invalid/v1"

        def factory(**_kwargs):
            raise RuntimeError(f"failed with {secret} at {url}")

        with self.assertRaises(DirectLLMAPIError) as caught:
            create_openai_client(
                APISettings(api_key_env="KEY", base_url_env="URL"),
                environ={"KEY": secret, "URL": url},
                client_factory=factory,
            )
        rendered = str(caught.exception)
        self.assertNotIn(secret, rendered)
        self.assertNotIn(url, rendered)

    def test_exponential_retry_and_format_reminder_messages(self) -> None:
        fake = FakeClient(
            [
                HTTPError("server included secret-token", 429),
                HTTPError("server included secret-token", 503),
                response('{"future_trajectory":[]}'),
            ]
        )
        sleeps = []
        settings = APISettings(
            max_api_retries=3,
            initial_backoff_seconds=0.5,
            max_backoff_seconds=5.0,
        )
        result = complete_with_retries(
            fake,
            settings,
            system_prompt="system",
            user_prompt="observation",
            format_reminder="fixed reminder",
            sleep=sleeps.append,
        )
        self.assertEqual(result.api_attempts, 3)
        self.assertEqual(sleeps, [0.5, 1.0])
        messages = fake.endpoint.calls[-1]["messages"]
        self.assertEqual([item["role"] for item in messages], ["system", "user"])
        self.assertIn("observation", messages[-1]["content"])
        self.assertTrue(messages[-1]["content"].endswith("fixed reminder"))

    def test_nonretryable_error_is_sanitized(self) -> None:
        secret = "DO_NOT_LOG_API_SENTINEL"
        fake = FakeClient([HTTPError(f"unauthorized {secret}", 401)])
        with self.assertRaises(DirectLLMAPIError) as caught:
            complete_with_retries(
                fake,
                APISettings(max_api_retries=4),
                system_prompt="system",
                user_prompt="observation",
                sleep=lambda _delay: self.fail("401 must not be retried"),
            )
        self.assertEqual(caught.exception.attempts, 1)
        self.assertNotIn(secret, str(caught.exception))
        self.assertIn("HTTP 401", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
