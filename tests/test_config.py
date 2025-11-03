"""Unit tests for configuration classes."""

from unittest.mock import Mock, patch

import pytest
from strands.models.anthropic import AnthropicModel
from strands.models.gemini import GeminiModel
from strands.models.ollama import OllamaModel

from kubernetes_assistant.config import KubernetesAssistantConfig, ModelConfig


class TestModelConfigDefaults:
    """Test default values for ModelConfig."""

    def test_default_provider_is_ollama(self, monkeypatch):
        """Test that default provider is ollama."""
        # Clear any existing environment variables
        for key in [
            "LLM_PROVIDER",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
        ]:
            monkeypatch.delenv(key, raising=False)

        config = ModelConfig()
        assert config.provider == "ollama"

    def test_default_ollama_values(self, monkeypatch):
        """Test default Ollama configuration values."""
        # Clear any existing environment variables
        for key in ["OLLAMA_HOST", "OLLAMA_MODEL_ID"]:
            monkeypatch.delenv(key, raising=False)

        config = ModelConfig()
        assert config.ollama_host == "http://localhost:11434"
        assert config.ollama_model_id == "qwen3:latest"

    def test_default_anthropic_values(self, monkeypatch):
        """Test default Anthropic configuration values."""
        # Clear any existing environment variables
        for key in [
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_MODEL_ID",
            "ANTHROPIC_MAX_TOKENS",
            "ANTHROPIC_TEMPERATURE",
        ]:
            monkeypatch.delenv(key, raising=False)

        config = ModelConfig()
        assert config.anthropic_api_key is None
        assert config.anthropic_model_id == "claude-sonnet-4-20250514"
        assert config.anthropic_max_tokens == 4096
        assert config.anthropic_temperature == 0.3

    def test_default_gemini_values(self, monkeypatch):
        """Test default Gemini configuration values."""
        # Clear any existing environment variables
        for key in [
            "GEMINI_API_KEY",
            "GEMINI_MODEL_ID",
            "GEMINI_TEMPERATURE",
            "GEMINI_MAX_OUTPUT_TOKENS",
            "GEMINI_TOP_P",
            "GEMINI_TOP_K",
        ]:
            monkeypatch.delenv(key, raising=False)

        config = ModelConfig()
        assert config.gemini_api_key is None
        assert config.gemini_model_id == "gemini-2.5-flash"
        assert config.gemini_temperature == 0.7
        assert config.gemini_max_output_tokens == 2048
        assert config.gemini_top_p == 0.9
        assert config.gemini_top_k == 40


class TestModelConfigEnvironmentVariables:
    """Test environment variable override for ModelConfig."""

    def test_provider_from_env(self, monkeypatch):
        """Test provider can be set via LLM_PROVIDER environment variable."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        config = ModelConfig()
        assert config.provider == "anthropic"

    def test_ollama_host_from_env(self, monkeypatch):
        """Test Ollama host can be set via environment variable."""
        monkeypatch.setenv("OLLAMA_HOST", "http://custom-host:8080")
        config = ModelConfig()
        assert config.ollama_host == "http://custom-host:8080"

    def test_ollama_model_id_from_env(self, monkeypatch):
        """Test Ollama model ID can be set via environment variable."""
        monkeypatch.setenv("OLLAMA_MODEL_ID", "llama3:latest")
        config = ModelConfig()
        assert config.ollama_model_id == "llama3:latest"

    def test_anthropic_api_key_from_env(self, monkeypatch):
        """Test Anthropic API key can be set via environment variable."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test123")
        config = ModelConfig()
        assert config.anthropic_api_key == "sk-ant-test123"

    def test_anthropic_model_id_from_env(self, monkeypatch):
        """Test Anthropic model ID can be set via environment variable."""
        monkeypatch.setenv("ANTHROPIC_MODEL_ID", "claude-opus-4-20250514")
        config = ModelConfig()
        assert config.anthropic_model_id == "claude-opus-4-20250514"

    def test_anthropic_max_tokens_from_env(self, monkeypatch):
        """Test Anthropic max tokens can be set via environment variable."""
        monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "8192")
        config = ModelConfig()
        assert config.anthropic_max_tokens == 8192

    def test_anthropic_temperature_from_env(self, monkeypatch):
        """Test Anthropic temperature can be set via environment variable."""
        monkeypatch.setenv("ANTHROPIC_TEMPERATURE", "0.7")
        config = ModelConfig()
        assert config.anthropic_temperature == 0.7

    def test_gemini_api_key_from_env(self, monkeypatch):
        """Test Gemini API key can be set via environment variable."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
        config = ModelConfig()
        assert config.gemini_api_key == "test-gemini-key"

    def test_gemini_model_id_from_env(self, monkeypatch):
        """Test Gemini model ID can be set via environment variable."""
        monkeypatch.setenv("GEMINI_MODEL_ID", "gemini-2.0-pro")
        config = ModelConfig()
        assert config.gemini_model_id == "gemini-2.0-pro"

    def test_gemini_temperature_from_env(self, monkeypatch):
        """Test Gemini temperature can be set via environment variable."""
        monkeypatch.setenv("GEMINI_TEMPERATURE", "0.5")
        config = ModelConfig()
        assert config.gemini_temperature == 0.5

    def test_gemini_max_output_tokens_from_env(self, monkeypatch):
        """Test Gemini max output tokens can be set via environment variable."""
        monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "4096")
        config = ModelConfig()
        assert config.gemini_max_output_tokens == 4096

    def test_gemini_top_p_from_env(self, monkeypatch):
        """Test Gemini top_p can be set via environment variable."""
        monkeypatch.setenv("GEMINI_TOP_P", "0.95")
        config = ModelConfig()
        assert config.gemini_top_p == 0.95

    def test_gemini_top_k_from_env(self, monkeypatch):
        """Test Gemini top_k can be set via environment variable."""
        monkeypatch.setenv("GEMINI_TOP_K", "50")
        config = ModelConfig()
        assert config.gemini_top_k == 50


class TestModelConfigCreateModelOllama:
    """Test create_model method for Ollama provider."""

    @patch("kubernetes_assistant.config.OllamaModel")
    def test_create_ollama_model_with_defaults(self, mock_ollama_class, monkeypatch):
        """Test creating Ollama model with default configuration."""
        # Clear environment variables
        for key in ["OLLAMA_HOST", "OLLAMA_MODEL_ID", "LLM_PROVIDER"]:
            monkeypatch.delenv(key, raising=False)

        mock_model = Mock()
        mock_ollama_class.return_value = mock_model

        config = ModelConfig()
        result = config.create_model()

        mock_ollama_class.assert_called_once_with(
            host="http://localhost:11434",
            model_id="qwen3:latest",
        )
        assert result == mock_model

    @patch("kubernetes_assistant.config.OllamaModel")
    def test_create_ollama_model_with_custom_config(self, mock_ollama_class, monkeypatch):
        """Test creating Ollama model with custom configuration."""
        monkeypatch.setenv("OLLAMA_HOST", "http://custom:9999")
        monkeypatch.setenv("OLLAMA_MODEL_ID", "custom-model:v1")

        mock_model = Mock()
        mock_ollama_class.return_value = mock_model

        config = ModelConfig()
        result = config.create_model()

        mock_ollama_class.assert_called_once_with(
            host="http://custom:9999",
            model_id="custom-model:v1",
        )
        assert result == mock_model

    def test_create_ollama_model_returns_correct_type(self, monkeypatch):
        """Test that creating Ollama model returns OllamaModel instance."""
        # Clear environment variables
        for key in ["LLM_PROVIDER"]:
            monkeypatch.delenv(key, raising=False)

        config = ModelConfig()
        result = config.create_model()

        assert isinstance(result, OllamaModel)


class TestModelConfigCreateModelAnthropic:
    """Test create_model method for Anthropic provider."""

    @patch("kubernetes_assistant.config.AnthropicModel")
    def test_create_anthropic_model_with_api_key(self, mock_anthropic_class, monkeypatch):
        """Test creating Anthropic model with API key."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test123")

        mock_model = Mock()
        mock_anthropic_class.return_value = mock_model

        config = ModelConfig()
        result = config.create_model()

        mock_anthropic_class.assert_called_once_with(
            client_args={"api_key": "sk-ant-test123"},
            model_id="claude-sonnet-4-20250514",
            max_tokens=4096,
            params={"temperature": 0.3},
        )
        assert result == mock_model

    @patch("kubernetes_assistant.config.AnthropicModel")
    def test_create_anthropic_model_with_custom_config(self, mock_anthropic_class, monkeypatch):
        """Test creating Anthropic model with custom configuration."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-custom")
        monkeypatch.setenv("ANTHROPIC_MODEL_ID", "claude-opus-4-20250514")
        monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "8192")
        monkeypatch.setenv("ANTHROPIC_TEMPERATURE", "0.7")

        mock_model = Mock()
        mock_anthropic_class.return_value = mock_model

        config = ModelConfig()
        result = config.create_model()

        mock_anthropic_class.assert_called_once_with(
            client_args={"api_key": "sk-ant-custom"},
            model_id="claude-opus-4-20250514",
            max_tokens=8192,
            params={"temperature": 0.7},
        )
        assert result == mock_model

    def test_create_anthropic_model_without_api_key_raises_error(self, monkeypatch):
        """Test that creating Anthropic model without API key raises ValueError."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        config = ModelConfig()

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is required"):
            config.create_model()

    def test_create_anthropic_model_with_none_api_key_raises_error(self, monkeypatch):
        """Test that creating Anthropic model with None API key raises ValueError."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        # Explicitly set to None through the config
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        config = ModelConfig()
        config.anthropic_api_key = None

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is required"):
            config.create_model()

    def test_create_anthropic_model_returns_correct_type(self, monkeypatch):
        """Test that creating Anthropic model returns AnthropicModel instance."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test123")

        config = ModelConfig()
        result = config.create_model()

        assert isinstance(result, AnthropicModel)


class TestModelConfigCreateModelGemini:
    """Test create_model method for Gemini provider."""

    @patch("kubernetes_assistant.config.GeminiModel")
    def test_create_gemini_model_with_api_key(self, mock_gemini_class, monkeypatch):
        """Test creating Gemini model with API key."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

        mock_model = Mock()
        mock_gemini_class.return_value = mock_model

        config = ModelConfig()
        result = config.create_model()

        mock_gemini_class.assert_called_once_with(
            client_args={"api_key": "test-gemini-key"},
            model_id="gemini-2.5-flash",
            params={
                "temperature": 0.7,
                "max_output_tokens": 2048,
                "top_p": 0.9,
                "top_k": 40,
            },
        )
        assert result == mock_model

    @patch("kubernetes_assistant.config.GeminiModel")
    def test_create_gemini_model_with_custom_config(self, mock_gemini_class, monkeypatch):
        """Test creating Gemini model with custom configuration."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "custom-key")
        monkeypatch.setenv("GEMINI_MODEL_ID", "gemini-2.0-pro")
        monkeypatch.setenv("GEMINI_TEMPERATURE", "0.5")
        monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "4096")
        monkeypatch.setenv("GEMINI_TOP_P", "0.95")
        monkeypatch.setenv("GEMINI_TOP_K", "50")

        mock_model = Mock()
        mock_gemini_class.return_value = mock_model

        config = ModelConfig()
        result = config.create_model()

        mock_gemini_class.assert_called_once_with(
            client_args={"api_key": "custom-key"},
            model_id="gemini-2.0-pro",
            params={
                "temperature": 0.5,
                "max_output_tokens": 4096,
                "top_p": 0.95,
                "top_k": 50,
            },
        )
        assert result == mock_model

    def test_create_gemini_model_without_api_key_raises_error(self, monkeypatch):
        """Test that creating Gemini model without API key raises ValueError."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        config = ModelConfig()

        with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
            config.create_model()

    def test_create_gemini_model_with_none_api_key_raises_error(self, monkeypatch):
        """Test that creating Gemini model with None API key raises ValueError."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        config = ModelConfig()
        config.gemini_api_key = None

        with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
            config.create_model()

    def test_create_gemini_model_returns_correct_type(self, monkeypatch):
        """Test that creating Gemini model returns GeminiModel instance."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        config = ModelConfig()
        result = config.create_model()

        assert isinstance(result, GeminiModel)


class TestModelConfigCreateModelErrors:
    """Test error handling in create_model method."""

    def test_create_model_with_invalid_provider_raises_validation_error(self, monkeypatch):
        """Test that invalid provider raises ValidationError at config creation time.

        Note: Pydantic validates the provider field using Literal type at initialization,
        so we never reach create_model() with an invalid provider.
        """
        from pydantic_core import ValidationError

        monkeypatch.setenv("LLM_PROVIDER", "unknown_provider")

        with pytest.raises(
            ValidationError, match="Input should be 'ollama', 'anthropic' or 'gemini'"
        ):
            ModelConfig()


class TestKubernetesAssistantConfigDefaults:
    """Test default values for KubernetesAssistantConfig."""

    def test_default_cluster_name(self, monkeypatch):
        """Test default cluster name."""
        monkeypatch.delenv("CLUSTER_NAME", raising=False)
        config = KubernetesAssistantConfig()
        assert config.cluster_name == "The Cluster"

    def test_default_agent_name(self, monkeypatch):
        """Test default agent name."""
        monkeypatch.delenv("AGENT_NAME", raising=False)
        config = KubernetesAssistantConfig()
        assert config.agent_name == "KubeBot"

    def test_default_agent_role(self, monkeypatch):
        """Test default agent role."""
        monkeypatch.delenv("AGENT_ROLE", raising=False)
        config = KubernetesAssistantConfig()
        assert config.agent_role == "intern system administrator"

    def test_default_kubeconfig_path(self, monkeypatch):
        """Test default kubeconfig path."""
        monkeypatch.delenv("KUBE_CONFIG_PATH", raising=False)
        config = KubernetesAssistantConfig()
        assert config.kubeconfig_path == "./config/k3s.yaml"

    def test_default_discord_token(self, monkeypatch):
        """Test default discord token is None."""
        monkeypatch.delenv("DISCORD_TOKEN", raising=False)
        config = KubernetesAssistantConfig()
        assert config.discord_token is None

    def test_default_config_dir(self, monkeypatch):
        """Test default config directory."""
        monkeypatch.delenv("CONFIG_DIR", raising=False)
        config = KubernetesAssistantConfig()
        assert config.config_dir == "./config"

    def test_default_prometheus_url(self, monkeypatch):
        """Test default prometheus URL is None."""
        monkeypatch.delenv("PROMETHEUS_URL", raising=False)
        config = KubernetesAssistantConfig()
        assert config.prometheus_url is None

    def test_default_llm_config_is_model_config(self, monkeypatch):
        """Test that llm_config is a ModelConfig instance."""
        config = KubernetesAssistantConfig()
        assert isinstance(config.llm_config, ModelConfig)


class TestKubernetesAssistantConfigEnvironmentVariables:
    """Test environment variable override for KubernetesAssistantConfig."""

    def test_cluster_name_from_env(self, monkeypatch):
        """Test cluster name can be set via environment variable."""
        monkeypatch.setenv("CLUSTER_NAME", "Production Cluster")
        config = KubernetesAssistantConfig()
        assert config.cluster_name == "Production Cluster"

    def test_agent_name_from_env(self, monkeypatch):
        """Test agent name can be set via environment variable."""
        monkeypatch.setenv("AGENT_NAME", "CustomBot")
        config = KubernetesAssistantConfig()
        assert config.agent_name == "CustomBot"

    def test_agent_role_from_env(self, monkeypatch):
        """Test agent role can be set via environment variable."""
        monkeypatch.setenv("AGENT_ROLE", "senior DevOps engineer")
        config = KubernetesAssistantConfig()
        assert config.agent_role == "senior DevOps engineer"

    def test_kubeconfig_path_from_env(self, monkeypatch):
        """Test kubeconfig path can be set via environment variable."""
        monkeypatch.setenv("KUBE_CONFIG_PATH", "/custom/path/kubeconfig.yaml")
        config = KubernetesAssistantConfig()
        assert config.kubeconfig_path == "/custom/path/kubeconfig.yaml"

    def test_discord_token_from_env(self, monkeypatch):
        """Test discord token can be set via environment variable."""
        monkeypatch.setenv("DISCORD_TOKEN", "my-discord-token-123")
        config = KubernetesAssistantConfig()
        assert config.discord_token == "my-discord-token-123"

    def test_config_dir_from_env(self, monkeypatch):
        """Test config directory can be set via environment variable."""
        monkeypatch.setenv("CONFIG_DIR", "/etc/k8s-assistant")
        config = KubernetesAssistantConfig()
        assert config.config_dir == "/etc/k8s-assistant"

    def test_prometheus_url_from_env(self, monkeypatch):
        """Test prometheus URL can be set via environment variable."""
        monkeypatch.setenv("PROMETHEUS_URL", "http://prometheus:9090")
        config = KubernetesAssistantConfig()
        assert config.prometheus_url == "http://prometheus:9090"


class TestKubernetesAssistantConfigNestedLLMConfig:
    """Test nested llm_config in KubernetesAssistantConfig."""

    def test_llm_config_inherits_environment_variables(self, monkeypatch):
        """Test that nested llm_config respects environment variables."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("ANTHROPIC_MODEL_ID", "claude-opus-4-20250514")

        config = KubernetesAssistantConfig()

        assert config.llm_config.provider == "anthropic"
        assert config.llm_config.anthropic_api_key == "test-key"
        assert config.llm_config.anthropic_model_id == "claude-opus-4-20250514"

    @patch("kubernetes_assistant.config.AnthropicModel")
    def test_llm_config_create_model_works(self, mock_anthropic_class, monkeypatch):
        """Test that create_model can be called on nested llm_config."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        mock_model = Mock()
        mock_anthropic_class.return_value = mock_model

        config = KubernetesAssistantConfig()
        result = config.llm_config.create_model()

        assert result == mock_model
        mock_anthropic_class.assert_called_once()


class TestKubernetesAssistantConfigIntegration:
    """Integration tests for KubernetesAssistantConfig."""

    def test_full_config_with_all_env_vars(self, monkeypatch):
        """Test full configuration with all environment variables set."""
        # Set all environment variables
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("ANTHROPIC_MODEL_ID", "claude-opus-4-20250514")
        monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "8192")
        monkeypatch.setenv("ANTHROPIC_TEMPERATURE", "0.5")
        monkeypatch.setenv("CLUSTER_NAME", "Production")
        monkeypatch.setenv("AGENT_NAME", "ProdBot")
        monkeypatch.setenv("AGENT_ROLE", "senior administrator")
        monkeypatch.setenv("KUBE_CONFIG_PATH", "/prod/kubeconfig")
        monkeypatch.setenv("DISCORD_TOKEN", "discord-token")
        monkeypatch.setenv("CONFIG_DIR", "/etc/config")
        monkeypatch.setenv("PROMETHEUS_URL", "http://prom:9090")

        config = KubernetesAssistantConfig()

        # Verify main config
        assert config.cluster_name == "Production"
        assert config.agent_name == "ProdBot"
        assert config.agent_role == "senior administrator"
        assert config.kubeconfig_path == "/prod/kubeconfig"
        assert config.discord_token == "discord-token"
        assert config.config_dir == "/etc/config"
        assert config.prometheus_url == "http://prom:9090"

        # Verify nested LLM config
        assert config.llm_config.provider == "anthropic"
        assert config.llm_config.anthropic_api_key == "sk-ant-test"
        assert config.llm_config.anthropic_model_id == "claude-opus-4-20250514"
        assert config.llm_config.anthropic_max_tokens == 8192
        assert config.llm_config.anthropic_temperature == 0.5

    def test_minimal_config_with_defaults(self, monkeypatch):
        """Test minimal configuration using all defaults."""
        # Clear all optional environment variables
        env_vars = [
            "LLM_PROVIDER",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "CLUSTER_NAME",
            "AGENT_NAME",
            "AGENT_ROLE",
            "KUBE_CONFIG_PATH",
            "DISCORD_TOKEN",
            "CONFIG_DIR",
            "PROMETHEUS_URL",
        ]
        for var in env_vars:
            monkeypatch.delenv(var, raising=False)

        config = KubernetesAssistantConfig()

        # Verify defaults
        assert config.cluster_name == "The Cluster"
        assert config.agent_name == "KubeBot"
        assert config.agent_role == "intern system administrator"
        assert config.kubeconfig_path == "./config/k3s.yaml"
        assert config.discord_token is None
        assert config.config_dir == "./config"
        assert config.prometheus_url is None
        assert config.llm_config.provider == "ollama"
