# accuralai-ollama

`accuralai-ollama` provides an `ollama` backend implementation for `accuralai-core`, enabling local model inference through the Ollama service. Once installed, you can configure the core orchestrator to route requests to Ollama by setting the backend plugin to `ollama`.

## Installation

```bash
pip install accuralai-core accuralai-ollama
```

## Configuration

Add an Ollama backend block to your AccuralAI configuration (e.g., `~/.accuralai/core.toml`):

```toml
[backends.ollama]
plugin = "ollama"
[backends.ollama.options]
model = "gemma3:4b"
host = "http://127.0.0.1:11434"
timeout_s = 60
keep_alive = "5m"
```

Then run:

```bash
accuralai-core generate --prompt "Explain quantum entanglement" --route ollama
```

The backend will call Ollama’s `/api/generate` endpoint, return the completion, and populate usage metadata based on the server’s response.
