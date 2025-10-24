"""Ollama backend adapter implementation."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field, ValidationError

from accuralai_core.config.schema import BackendSettings
from accuralai_core.contracts.errors import BackendError, ConfigurationError
from accuralai_core.contracts.models import GenerateRequest, GenerateResponse, Usage
from accuralai_core.contracts.protocols import Backend

LOGGER = logging.getLogger("accuralai.ollama")

class OllamaOptions(BaseModel):
    """Configuration for connecting to the Ollama server."""

    model: str = "llama3"
    host: str = "http://127.0.0.1:11434"
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1)
    timeout_s: float = Field(default=60.0, ge=1.0)
    keep_alive: str | None = Field(default=None)
    extra: Dict[str, Any] = Field(default_factory=dict)


class OllamaClient:
    """HTTP client wrapper for Ollama interactions."""

    def __init__(self, options: OllamaOptions) -> None:
        self._options = options
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure we have a valid HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._options.host, 
                timeout=self._options.timeout_s,
                limits=httpx.Limits(max_keepalive_connections=2, max_connections=5),
                http2=False,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client safely."""
        if self._client is not None:
            try:
                if not self._client.is_closed:
                    await self._client.aclose()
            except Exception as e:
                LOGGER.warning("Error closing Ollama client: %s", e)
            finally:
                self._client = None

    async def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        client = await self._ensure_client()
        try:
            response = await client.post("/api/generate", json=payload)
            response.raise_for_status()
            try:
                return response.json()
            except json.JSONDecodeError:
                return self._parse_streaming_json(response)
        except Exception as e:
            if "Event loop is closed" in str(e):
                LOGGER.warning("Event loop error detected, recreating client")
                await self.close()
                client = await self._ensure_client()
                response = await client.post("/api/generate", json=payload)
                response.raise_for_status()
                try:
                    return response.json()
                except json.JSONDecodeError:
                    return self._parse_streaming_json(response)
            raise

    def _parse_streaming_json(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle Ollama's newline-delimited JSON streaming format."""

        text = response.text.strip()
        if not text:
            raise BackendError("Empty response from Ollama")

        chunks: list[Dict[str, Any]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise BackendError(f"Failed to parse Ollama stream chunk: {error}") from error

        if not chunks:
            raise BackendError("Ollama stream contained no JSON chunks")

        aggregated: Dict[str, Any] = {}
        responses: list[str] = []
        for chunk in chunks:
            if "response" in chunk and chunk["response"]:
                responses.append(chunk["response"])
            if "output" in chunk and chunk["output"]:
                responses.append(chunk["output"])

            # Preserve the latest values for metadata style fields.
            for key in ("model", "created_at", "context", "done", "done_reason", "metrics", "total_duration", "usage"):
                if key in chunk and chunk[key] is not None:
                    aggregated[key] = chunk[key]

        aggregated.setdefault("done", True)
        aggregated.setdefault("done_reason", "stop")
        aggregated.setdefault("metrics", {})
        aggregated.setdefault("total_duration", 0)
        aggregated.setdefault("usage", {})
        aggregated["response"] = "".join(responses)

        return aggregated


@dataclass(slots=True)
class OllamaBackend(Backend):
    """Backend protocol implementation delegating to Ollama."""

    options: OllamaOptions
    client: OllamaClient = field(init=False)

    def __post_init__(self) -> None:
        self.client = OllamaClient(self.options)

    async def generate(self, request: GenerateRequest, *, routed_to: str) -> GenerateResponse:
        payload = self._build_payload(request)
        try:
            result = await self.client.generate(payload)
        except (httpx.HTTPError, asyncio.TimeoutError, RuntimeError) as error:
            if "Event loop is closed" in str(error):
                LOGGER.warning("Event loop error during Ollama request, attempting recovery")
                try:
                    await self.client.close()
                    result = await self.client.generate(payload)
                except Exception as retry_error:
                    raise BackendError(f"Ollama request failed after retry: {retry_error}", cause=retry_error) from retry_error
            else:
                raise BackendError(f"Ollama request failed: {error}", cause=error) from error

        return self._build_response(request, result)

    def _build_payload(self, request: GenerateRequest) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.options.model,
            "prompt": request.prompt,
            "system": request.system_prompt,
            "options": {
                **({"temperature": self.options.temperature} if self.options.temperature is not None else {}),
                **({"top_p": self.options.top_p} if self.options.top_p is not None else {}),
                **({"num_predict": self.options.max_tokens} if self.options.max_tokens is not None else {}),
                **request.parameters,
            },
            "keep_alive": self.options.keep_alive,
        }
        body["options"].update(self.options.extra)
        body = {key: value for key, value in body.items() if value is not None}
        if request.tools:
            body["tools"] = request.tools
        return body

    def _build_response(self, request: GenerateRequest, result: Mapping[str, Any]) -> GenerateResponse:
        output_text = result.get("response") or result.get("output") or ""
        tool_calls = self._extract_tool_calls(result)
        
        if tool_calls:
            finish_reason = "stop"
            if not output_text:
                output_text = "[tool-call]"
        else:
            finish_reason = result.get("done_reason", "stop") or "stop"
        
        metadata = {
            "model": result.get("model"),
            "created_at": result.get("created_at"),
            "metrics": result.get("metrics") or {},
            "tool_calls": tool_calls,
        }
        usage_data = result.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(usage_data.get("prompt_tokens", 0)),
            completion_tokens=int(usage_data.get("completion_tokens", 0)),
            total_tokens=int(usage_data.get("total_tokens", 0)),
            extra={key: value for key, value in usage_data.items() if key not in {"prompt_tokens", "completion_tokens", "total_tokens"}},
        )
        return GenerateResponse(
            id=uuid4(),
            request_id=request.id,
            output_text=output_text,
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=int((result.get("total_duration", 0) or 0) / 1_000_000),
            metadata=metadata,
            validator_events=[],
        )

    def _extract_tool_calls(self, result: Mapping[str, Any]) -> list[Dict[str, Any]]:
        """Extract tool calls from Ollama response."""
        tool_calls: list[Dict[str, Any]] = []
        response_text = result.get("response") or result.get("output") or ""
        
        try:
            if response_text.strip().startswith('{') and response_text.strip().endswith('}'):
                parsed = json.loads(response_text)
                if isinstance(parsed, dict) and "tool_calls" in parsed:
                    tool_calls_data = parsed["tool_calls"]
                    if isinstance(tool_calls_data, list):
                        for call in tool_calls_data:
                            if isinstance(call, dict) and "name" in call:
                                tool_calls.append({
                                    "id": f"call_{uuid4().hex}",
                                    "name": str(call["name"]),
                                    "arguments": call.get("arguments", {}),
                                })
        except json.JSONDecodeError:
            pass
        
        if "tool_calls" in result:
            tool_calls_data = result["tool_calls"]
            if isinstance(tool_calls_data, list):
                for call in tool_calls_data:
                    if isinstance(call, dict) and "name" in call:
                        tool_calls.append({
                            "id": f"call_{uuid4().hex}",
                            "name": str(call["name"]),
                            "arguments": call.get("arguments", {}),
                        })
        
        return tool_calls

    async def aclose(self) -> None:
        """Clean up resources."""
        try:
            await self.client.close()
        except Exception as e:
            LOGGER.warning("Error closing Ollama backend: %s", e)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()


async def build_ollama_backend(
    *,
    config: BackendSettings | Mapping[str, Any] | None = None,
    backend_id: str | None = None,
    **_: Any,
) -> Backend:
    """Entry point registered with accuralai-core to build the Ollama backend."""
    options_data: Mapping[str, Any] | None = None
    if isinstance(config, BackendSettings):
        options_data = config.options or {}
    elif isinstance(config, Mapping):
        options_data = dict(config)

    try:
        options = OllamaOptions.model_validate(options_data or {})
    except ValidationError as error:
        raise ConfigurationError(f"Invalid Ollama backend configuration: {error}") from error

    LOGGER.info("Configuring Ollama backend '%s' targeting %s", backend_id or "ollama", options.host)
    return OllamaBackend(options=options)
