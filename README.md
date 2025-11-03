# Kubernetes Assistant

[![PyPI - Version](https://img.shields.io/pypi/v/kubernetes-assistant.svg)](https://pypi.org/project/kubernetes-assistant)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/kubernetes-assistant.svg)](https://pypi.org/project/kubernetes-assistant)

An AI powered assistant that helps manage and monitor Kubernetes clusters. The assistant provides intelligent cluster insights and operational support through natural language interactions.

-----

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [License](#license)

## Installation

```console
pip install kubernetes-assistant
```

## Configuration

Configure the assistant using environment variables or a `.env` file:

### LLM Provider Configuration

| Environment Variable | Description | Default | Required |
|---------------------|-------------|---------|----------|
| `LLM_PROVIDER` | LLM provider to use (`ollama`, `anthropic`, or `gemini`) | `ollama` | No |

### Ollama Configuration

| Environment Variable | Description | Default | Required |
|---------------------|-------------|---------|----------|
| `OLLAMA_HOST` | Ollama API endpoint | `http://localhost:11434` | No |
| `OLLAMA_MODEL_ID` | Ollama model identifier | `qwen3:latest` | No |

### Anthropic Configuration

| Environment Variable | Description | Default | Required |
|---------------------|-------------|---------|----------|
| `ANTHROPIC_API_KEY` | Anthropic API key | - | Yes (when using `anthropic` provider) |
| `ANTHROPIC_MODEL_ID` | Anthropic model identifier | `claude-sonnet-4-20250514` | No |
| `ANTHROPIC_MAX_TOKENS` | Maximum tokens for Anthropic responses | `4096` | No |
| `ANTHROPIC_TEMPERATURE` | Temperature for Anthropic model | `0.3` | No |

### Gemini Configuration

| Environment Variable | Description | Default | Required |
|---------------------|-------------|---------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | - | Yes (when using `gemini` provider) |
| `GEMINI_MODEL_ID` | Gemini model identifier | `gemini-2.5-flash` | No |
| `GEMINI_TEMPERATURE` | Temperature for Gemini model | `0.7` | No |
| `GEMINI_MAX_OUTPUT_TOKENS` | Maximum output tokens for Gemini | `2048` | No |
| `GEMINI_TOP_P` | Top-p sampling for Gemini | `0.9` | No |
| `GEMINI_TOP_K` | Top-k sampling for Gemini | `40` | No |

### Application Configuration

| Environment Variable | Description | Default | Required |
|---------------------|-------------|---------|----------|
| `CLUSTER_NAME` | Display name for your cluster | `The Cluster` | No |
| `AGENT_NAME` | Bot's display name | `KubeBot` | No |
| `AGENT_ROLE` | Bot's persona/role description | `intern system administrator` | No |
| `CUSTOM_AGENT_PROMPT` | Override the default agent prompt entirely. When set, this replaces the generated prompt. | - | No |
| `KUBE_CONFIG_PATH` | Path to kubeconfig file | `./config/k3s.yaml` | No |
| `CONFIG_DIR` | Configuration directory path | `./config` | No |
| `PROMETHEUS_URL` | Prometheus server endpoint URL | - | No |
| `DISCORD_TOKEN` | Discord bot token (only needed when running as Discord bot) | - | No |

## License

`kubernetes-assistant` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

