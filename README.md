# Kubernetes Assistant

[![PyPI - Version](https://img.shields.io/pypi/v/kubernetes-assistant.svg)](https://pypi.org/project/kubernetes-assistant)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/kubernetes-assistant.svg)](https://pypi.org/project/kubernetes-assistant)

A Discord bot powered by LLMs that helps manage and monitor Kubernetes clusters. The assistant provides intelligent cluster insights and operational support through natural language interactions.

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

| Environment Variable | Description | Default | Required |
|---------------------|-------------|---------|----------|
| `MODEL_HOST` | LLM API endpoint (Ollama-compatible) | `http://localhost:11434` | No |
| `MODEL_ID` | Model identifier to use | `qwen3:latest` | No |
| `CLUSTER_NAME` | Display name for your cluster | `The Cluster` | No |
| `AGENT_NAME` | Bot's display name | `KubeBot` | No |
| `AGENT_ROLE` | Bot's persona/role description | `intern system administrator` | No |
| `KUBE_CONFIG_PATH` | Path to kubeconfig file | `./config/k3s.yaml` | No |
| `CONFIG_DIR` | Configuration directory path | `./config` | No |
| `DISCORD_TOKEN` | Discord bot token | - | **Yes** |

## License

`kubernetes-assistant` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

