from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings
from strands.models.anthropic import AnthropicModel
from strands.models.gemini import GeminiModel
from strands.models.model import Model
from strands.models.ollama import OllamaModel


class ModelConfig(BaseSettings):
    # Provider selection
    provider: Literal["ollama", "anthropic", "gemini"] = Field(
        default="ollama", alias="LLM_PROVIDER"
    )

    # Ollama configuration
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    ollama_model_id: str = Field(default="qwen3:latest", alias="OLLAMA_MODEL_ID")

    # Anthropic configuration
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model_id: str = Field(default="claude-sonnet-4-20250514", alias="ANTHROPIC_MODEL_ID")
    anthropic_max_tokens: int = Field(default=4096, alias="ANTHROPIC_MAX_TOKENS")
    anthropic_temperature: float = Field(default=0.3, alias="ANTHROPIC_TEMPERATURE")

    # Gemini configuration
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model_id: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL_ID")
    gemini_temperature: float = Field(default=0.7, alias="GEMINI_TEMPERATURE")
    gemini_max_output_tokens: int = Field(default=2048, alias="GEMINI_MAX_OUTPUT_TOKENS")
    gemini_top_p: float = Field(default=0.9, alias="GEMINI_TOP_P")
    gemini_top_k: int = Field(default=40, alias="GEMINI_TOP_K")

    def create_model(self) -> Model:
        """Create and return the appropriate model based on the provider setting."""
        if self.provider == "ollama":
            return OllamaModel(
                host=self.ollama_host,
                model_id=self.ollama_model_id,
            )
        elif self.provider == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is required when using Anthropic provider")
            return AnthropicModel(
                client_args={"api_key": self.anthropic_api_key},
                model_id=self.anthropic_model_id,
                max_tokens=self.anthropic_max_tokens,
                params={"temperature": self.anthropic_temperature},
            )
        elif self.provider == "gemini":
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is required when using Gemini provider")
            return GeminiModel(
                client_args={"api_key": self.gemini_api_key},
                model_id=self.gemini_model_id,
                params={
                    "temperature": self.gemini_temperature,
                    "max_output_tokens": self.gemini_max_output_tokens,
                    "top_p": self.gemini_top_p,
                    "top_k": self.gemini_top_k,
                },
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")


class KubernetesAssistantConfig(BaseSettings):
    llm_config: ModelConfig = Field(default_factory=ModelConfig)
    cluster_name: str = Field(default="The Cluster", alias="CLUSTER_NAME")
    agent_name: str = Field(default="KubeBot", alias="AGENT_NAME")
    agent_role: str = Field(default="intern system administrator", alias="AGENT_ROLE")
    custom_agent_prompt: str | None = Field(
        default=None,
        alias="CUSTOM_AGENT_PROMPT",
        description="Override the default agent prompt entirely. If set, this will be used instead of the generated prompt.",
    )
    kubeconfig_path: str = Field(default="./config/k3s.yaml", alias="KUBE_CONFIG_PATH")
    discord_token: str | None = Field(default=None, alias="DISCORD_TOKEN")
    config_dir: str = Field(default="./config", alias="CONFIG_DIR")
    prometheus_url: str | None = Field(default=None, alias="PROMETHEUS_URL")
