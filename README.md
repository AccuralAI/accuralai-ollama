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
options.model = "llama3"
options.host = "http://127.0.0.1:11434"
options.timeout_s = 60
```

Then run:

```bash
accuralai-core generate --prompt "Explain quantum entanglement" --route ollama
```

The backend will call Ollama’s `/api/generate` endpoint, return the completion, and populate usage metadata based on the server’s response.
