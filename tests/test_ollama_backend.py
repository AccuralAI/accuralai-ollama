from __future__ import annotations

import pytest

from accuralai_core.contracts.errors import ConfigurationError
from accuralai_core.contracts.models import GenerateRequest

from accuralai_ollama import backend as backend_module
from accuralai_ollama.backend import OllamaBackend, OllamaOptions, build_ollama_backend


class DummyClient:
    instance: "DummyClient | None" = None

    def __init__(self, options: OllamaOptions) -> None:
        self.options = options
        self.calls: list[dict[str, object]] = []
        DummyClient.instance = self

    async def generate(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(payload)
        return {
            "model": self.options.model,
            "response": "dummy completion",
            "done_reason": "stop",
            "total_duration": 5_000_000,
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "total_tokens": 12,
                "gpu": "on",
            },
        }

    async def close(self) -> None:  # pragma: no cover - nothing to clean up
        return None


@pytest.mark.anyio("asyncio")
async def test_ollama_backend_generates_response(monkeypatch):
    monkeypatch.setattr(backend_module, "OllamaClient", DummyClient)

    backend = OllamaBackend(
        options=OllamaOptions(model="custom", temperature=0.2, extra={"stop": ["\n"]})
    )
    request = GenerateRequest(
        prompt="Hello",
        system_prompt="You are friendly.",
        parameters={"temperature": 0.5},
    )

    response = await backend.generate(request, routed_to="ollama")

    dummy = DummyClient.instance
    assert dummy is not None
    payload = dummy.calls[0]
    assert payload["model"] == "custom"
    assert payload["prompt"] == "Hello"
    assert payload["system"] == "You are friendly."
    assert payload["options"]["temperature"] == 0.5
    assert payload["options"]["stop"] == ["\n"]

    assert response.output_text == "dummy completion"
    assert response.metadata["model"] == "custom"
    assert response.usage.total_tokens == 12
    assert response.usage.extra["gpu"] == "on"
    assert response.latency_ms == 5


@pytest.mark.anyio("asyncio")
async def test_build_ollama_backend_from_settings(monkeypatch):
    monkeypatch.setattr(backend_module, "OllamaClient", DummyClient)

    backend = await build_ollama_backend(config={"model": "mistral", "host": "http://test"})
    assert isinstance(backend, OllamaBackend)
    assert backend.options.model == "mistral"
    assert backend.options.host == "http://test"


@pytest.mark.anyio("asyncio")
async def test_build_ollama_backend_invalid(monkeypatch):
    monkeypatch.setattr(backend_module, "OllamaClient", DummyClient)

    with pytest.raises(ConfigurationError):
        await build_ollama_backend(config={"max_tokens": 0})
