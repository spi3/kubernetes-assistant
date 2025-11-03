"""Unit tests for KubernetesAssistantAgent class."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from strands.models.model import Model

from kubernetes_assistant.config import KubernetesAssistantConfig
from kubernetes_assistant.kubernetes_assistant_agent import KubernetesAssistantAgent


@pytest.fixture
def mock_config(monkeypatch):
    """Create a mock KubernetesAssistantConfig for testing."""
    # Set environment variables for pydantic-settings
    monkeypatch.setenv("OLLAMA_HOST", "http://test-host:11434")
    monkeypatch.setenv("OLLAMA_MODEL_ID", "test-model:latest")
    monkeypatch.setenv("CLUSTER_NAME", "TestCluster")
    monkeypatch.setenv("AGENT_NAME", "TestBot")
    monkeypatch.setenv("AGENT_ROLE", "test administrator")
    monkeypatch.setenv("KUBE_CONFIG_PATH", "/test/path/kubeconfig.yaml")
    monkeypatch.setenv("DISCORD_TOKEN", "test-discord-token")
    monkeypatch.setenv("CONFIG_DIR", "/test/config")
    # Ensure PROMETHEUS_URL is not set for tests
    monkeypatch.delenv("PROMETHEUS_URL", raising=False)

    config = KubernetesAssistantConfig()
    return config


@pytest.fixture
def mock_model():
    """Create a mock Model for testing."""
    model = Mock(spec=Model)
    model.name = "test-model"
    return model


@pytest.fixture
def session_id():
    """Provide a test session ID."""
    return "test-session-123"


@pytest.fixture
def agent_instance(mock_config, mock_model, session_id):
    """Create a KubernetesAssistantAgent instance with mocked dependencies."""
    with patch("kubernetes_assistant.kubernetes_assistant_agent.FileSessionManager"):
        with patch("kubernetes_assistant.config.ModelConfig.create_model", return_value=mock_model):
            agent = KubernetesAssistantAgent(config=mock_config, session_id=session_id)
            return agent


class TestKubernetesAssistantAgentInit:
    """Test cases for KubernetesAssistantAgent initialization."""

    def test_init_stores_config(self, mock_config, mock_model, session_id):
        """Test that __init__ correctly stores the config."""
        with patch("kubernetes_assistant.kubernetes_assistant_agent.FileSessionManager"):
            with patch(
                "kubernetes_assistant.config.ModelConfig.create_model", return_value=mock_model
            ):
                agent = KubernetesAssistantAgent(config=mock_config, session_id=session_id)
                assert agent.config == mock_config

    def test_init_stores_model(self, mock_config, mock_model, session_id):
        """Test that __init__ correctly stores the model."""
        with patch("kubernetes_assistant.kubernetes_assistant_agent.FileSessionManager"):
            with patch(
                "kubernetes_assistant.config.ModelConfig.create_model", return_value=mock_model
            ):
                agent = KubernetesAssistantAgent(config=mock_config, session_id=session_id)
                assert agent.model == mock_model

    @patch("kubernetes_assistant.kubernetes_assistant_agent.Agent")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.MCPClient")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.SummarizingConversationManager")
    def test_init_creates_session_manager(
        self,
        mock_conv_manager,
        mock_mcp_client_class,
        mock_agent_class,
        mock_config,
        mock_model,
        session_id,
    ):
        """Test that __enter__ creates a FileSessionManager with correct parameters."""
        # Setup mocks
        mock_mcp_instance = MagicMock()
        mock_mcp_client_class.return_value = mock_mcp_instance
        mock_mcp_instance.list_tools_sync.return_value = []

        with patch(
            "kubernetes_assistant.kubernetes_assistant_agent.FileSessionManager"
        ) as mock_session_manager:
            with patch(
                "kubernetes_assistant.config.ModelConfig.create_model", return_value=mock_model
            ):
                agent = KubernetesAssistantAgent(config=mock_config, session_id=session_id)

                # Session manager should not be created yet
                mock_session_manager.assert_not_called()

            # Enter the context manager
            with agent:
                # Now session manager should be created
                mock_session_manager.assert_called_once_with(
                    session_id=session_id, storage_dir="/test/config/sessions"
                )

    @patch("kubernetes_assistant.kubernetes_assistant_agent.Agent")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.MCPClient")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.SummarizingConversationManager")
    def test_init_with_different_config_dir(
        self,
        mock_conv_manager,
        mock_mcp_client_class,
        mock_agent_class,
        mock_model,
        session_id,
        monkeypatch,
    ):
        """Test session manager path generation with different config directories."""
        monkeypatch.setenv("CONFIG_DIR", "/custom/config/path")
        monkeypatch.setenv("DISCORD_TOKEN", "test-token")

        custom_config = KubernetesAssistantConfig()

        # Setup mocks
        mock_mcp_instance = MagicMock()
        mock_mcp_client_class.return_value = mock_mcp_instance
        mock_mcp_instance.list_tools_sync.return_value = []

        with patch(
            "kubernetes_assistant.kubernetes_assistant_agent.FileSessionManager"
        ) as mock_session_manager:
            with patch(
                "kubernetes_assistant.config.ModelConfig.create_model", return_value=mock_model
            ):
                agent = KubernetesAssistantAgent(config=custom_config, session_id=session_id)

            # Enter the context manager to trigger session manager creation
            with agent:
                mock_session_manager.assert_called_once_with(
                    session_id=session_id, storage_dir="/custom/config/path/sessions"
                )


class TestKubernetesAssistantAgentRun:
    """Test cases for KubernetesAssistantAgent run method."""

    @patch("kubernetes_assistant.kubernetes_assistant_agent.FileSessionManager")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.Agent")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.MCPClient")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.stdio_client")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.SummarizingConversationManager")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.agent_prompt")
    def test_run_creates_mcp_client_with_correct_params(
        self,
        mock_agent_prompt,
        mock_conv_manager,
        mock_stdio_client,
        mock_mcp_client_class,
        mock_agent_class,
        mock_session_manager,
        mock_config,
        mock_model,
        session_id,
    ):
        """Test that MCPClient is created with correct StdioServerParameters."""
        # Setup mocks
        mock_mcp_instance = MagicMock()
        mock_mcp_client_class.return_value = mock_mcp_instance
        mock_mcp_instance.list_tools_sync.return_value = []

        mock_agent_instance = MagicMock()
        mock_agent_result = Mock()
        mock_agent_result.output = "test output"
        mock_agent_instance.return_value = mock_agent_result
        mock_agent_class.return_value = mock_agent_instance

        # Create agent with mocked dependencies
        with patch("kubernetes_assistant.config.ModelConfig.create_model", return_value=mock_model):
            agent = KubernetesAssistantAgent(config=mock_config, session_id=session_id)

        # Enter context manager and run
        with agent:
            _result = agent.run("test input")

        # Verify MCPClient was created
        mock_mcp_client_class.assert_called()

        # Verify the lambda passed to MCPClient calls stdio_client correctly
        client_factory = mock_mcp_client_class.call_args[0][0]
        client_factory()  # Execute the lambda

        # Check that stdio_client was called
        assert mock_stdio_client.called

    @patch("kubernetes_assistant.kubernetes_assistant_agent.FileSessionManager")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.Agent")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.MCPClient")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.stdio_client")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.SummarizingConversationManager")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.agent_prompt")
    def test_run_lists_tools_from_mcp_client(
        self,
        mock_agent_prompt,
        mock_conv_manager,
        mock_stdio_client,
        mock_mcp_client_class,
        mock_agent_class,
        mock_session_manager,
        mock_config,
        mock_model,
        session_id,
    ):
        """Test that entering context calls list_tools_sync on the MCP client."""
        # Setup mocks
        mock_mcp_instance = MagicMock()
        mock_mcp_client_class.return_value = mock_mcp_instance
        mock_tools = [Mock(name="tool1"), Mock(name="tool2")]
        mock_mcp_instance.list_tools_sync.return_value = mock_tools

        mock_agent_instance = MagicMock()
        mock_agent_result = Mock()
        mock_agent_result.output = "test output"
        mock_agent_instance.return_value = mock_agent_result
        mock_agent_class.return_value = mock_agent_instance

        # Create agent with mocked dependencies
        with patch("kubernetes_assistant.config.ModelConfig.create_model", return_value=mock_model):
            agent = KubernetesAssistantAgent(config=mock_config, session_id=session_id)

        # Enter context manager and run
        with agent:
            agent.run("test input")

        # Verify list_tools_sync was called
        mock_mcp_instance.list_tools_sync.assert_called_once()

    @patch("kubernetes_assistant.kubernetes_assistant_agent.FileSessionManager")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.Agent")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.MCPClient")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.stdio_client")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.SummarizingConversationManager")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.agent_prompt")
    def test_run_creates_agent_with_correct_parameters(
        self,
        mock_agent_prompt,
        mock_conv_manager,
        mock_stdio_client,
        mock_mcp_client_class,
        mock_agent_class,
        mock_session_manager,
        mock_config,
        mock_model,
        session_id,
    ):
        """Test that context manager creates an Agent with correct parameters."""
        # Setup mocks
        mock_mcp_instance = MagicMock()
        mock_mcp_client_class.return_value = mock_mcp_instance
        mock_tools = [Mock(name="tool1"), Mock(name="tool2")]
        mock_mcp_instance.list_tools_sync.return_value = mock_tools

        mock_prompt = "Test system prompt"
        mock_agent_prompt.return_value = mock_prompt

        mock_conv_manager_instance = MagicMock()
        mock_conv_manager.return_value = mock_conv_manager_instance

        mock_session_manager_instance = MagicMock()
        mock_session_manager.return_value = mock_session_manager_instance

        mock_agent_instance = MagicMock()
        mock_agent_result = Mock()
        mock_agent_result.output = "test output"
        mock_agent_instance.return_value = mock_agent_result
        mock_agent_class.return_value = mock_agent_instance

        # Create agent with mocked dependencies
        with patch("kubernetes_assistant.config.ModelConfig.create_model", return_value=mock_model):
            agent = KubernetesAssistantAgent(config=mock_config, session_id=session_id)

        # Enter context manager and run
        with agent:
            agent.run("test input")

        # Verify agent_prompt was called with correct parameters
        mock_agent_prompt.assert_called_once_with(
            agent.config.agent_name,
            agent.config.cluster_name,
            agent.config.agent_role,
        )

        # Verify SummarizingConversationManager was created without parameters
        mock_conv_manager.assert_called_once_with()

        # Verify Agent was created with correct parameters
        mock_agent_class.assert_called_once_with(
            model=agent.model,
            tools=mock_tools,
            system_prompt=mock_prompt,
            conversation_manager=mock_conv_manager_instance,
            session_manager=mock_session_manager_instance,
        )

    @patch("kubernetes_assistant.kubernetes_assistant_agent.FileSessionManager")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.Agent")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.MCPClient")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.stdio_client")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.SummarizingConversationManager")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.agent_prompt")
    def test_run_calls_agent_with_input(
        self,
        mock_agent_prompt,
        mock_conv_manager,
        mock_stdio_client,
        mock_mcp_client_class,
        mock_agent_class,
        mock_session_manager,
        mock_config,
        mock_model,
        session_id,
    ):
        """Test that run calls the agent with the provided input."""
        # Setup mocks
        mock_mcp_instance = MagicMock()
        mock_mcp_client_class.return_value = mock_mcp_instance
        mock_mcp_instance.list_tools_sync.return_value = []

        mock_agent_instance = MagicMock()
        mock_agent_result = Mock()
        mock_agent_result.output = "test output"
        mock_agent_instance.return_value = mock_agent_result
        mock_agent_class.return_value = mock_agent_instance

        test_input = "What is the status of my pods?"

        # Create agent with mocked dependencies
        with patch("kubernetes_assistant.config.ModelConfig.create_model", return_value=mock_model):
            agent = KubernetesAssistantAgent(config=mock_config, session_id=session_id)

        # Enter context manager and run
        with agent:
            _result = agent.run(test_input)

        # Verify the agent was called with the input
        mock_agent_instance.assert_called_once_with(test_input)

    @patch("kubernetes_assistant.kubernetes_assistant_agent.FileSessionManager")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.Agent")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.MCPClient")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.stdio_client")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.SummarizingConversationManager")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.agent_prompt")
    def test_run_returns_agent_result(
        self,
        mock_agent_prompt,
        mock_conv_manager,
        mock_stdio_client,
        mock_mcp_client_class,
        mock_agent_class,
        mock_session_manager,
        mock_config,
        mock_model,
        session_id,
    ):
        """Test that run returns the AgentResult from the agent call."""
        # Setup mocks
        mock_mcp_instance = MagicMock()
        mock_mcp_client_class.return_value = mock_mcp_instance
        mock_mcp_instance.list_tools_sync.return_value = []

        mock_agent_instance = MagicMock()
        expected_result = Mock()
        expected_result.output = "expected output"
        mock_agent_instance.return_value = expected_result
        mock_agent_class.return_value = mock_agent_instance

        # Create agent with mocked dependencies
        with patch("kubernetes_assistant.config.ModelConfig.create_model", return_value=mock_model):
            agent = KubernetesAssistantAgent(config=mock_config, session_id=session_id)

        # Enter context manager and run
        with agent:
            result = agent.run("test input")

        # Verify the result
        assert result == expected_result
        assert result.output == "expected output"

    @patch("kubernetes_assistant.kubernetes_assistant_agent.FileSessionManager")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.Agent")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.MCPClient")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.stdio_client")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.SummarizingConversationManager")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.agent_prompt")
    def test_run_uses_context_manager_for_mcp_client(
        self,
        mock_agent_prompt,
        mock_conv_manager,
        mock_stdio_client,
        mock_mcp_client_class,
        mock_agent_class,
        mock_session_manager,
        mock_config,
        mock_model,
        session_id,
    ):
        """Test that agent uses MCPClient as a context manager."""
        # Setup mocks
        mock_mcp_instance = MagicMock()
        mock_mcp_client_class.return_value = mock_mcp_instance
        mock_mcp_instance.list_tools_sync.return_value = []

        mock_agent_instance = MagicMock()
        mock_agent_result = Mock()
        mock_agent_result.output = "test output"
        mock_agent_instance.return_value = mock_agent_result
        mock_agent_class.return_value = mock_agent_instance

        # Create agent with mocked dependencies
        with patch("kubernetes_assistant.config.ModelConfig.create_model", return_value=mock_model):
            agent = KubernetesAssistantAgent(config=mock_config, session_id=session_id)

        # Use the agent with context manager
        with agent:
            agent.run("test input")

        # Verify context manager methods were called
        mock_mcp_instance.__enter__.assert_called_once()
        mock_mcp_instance.__exit__.assert_called_once()

    @patch("kubernetes_assistant.kubernetes_assistant_agent.FileSessionManager")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.Agent")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.MCPClient")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.stdio_client")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.SummarizingConversationManager")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.agent_prompt")
    def test_run_passes_kubeconfig_path_to_mcp(
        self,
        mock_agent_prompt,
        mock_conv_manager,
        mock_stdio_client,
        mock_mcp_client_class,
        mock_agent_class,
        mock_session_manager,
        mock_config,
        mock_model,
        session_id,
    ):
        """Test that agent passes the correct kubeconfig path to the MCP server."""
        # Setup mocks
        mock_mcp_instance = MagicMock()
        mock_mcp_client_class.return_value = mock_mcp_instance
        mock_mcp_instance.list_tools_sync.return_value = []

        mock_agent_instance = MagicMock()
        mock_agent_result = Mock()
        mock_agent_result.output = "test output"
        mock_agent_instance.return_value = mock_agent_result
        mock_agent_class.return_value = mock_agent_instance

        # Create agent with mocked dependencies
        with patch("kubernetes_assistant.config.ModelConfig.create_model", return_value=mock_model):
            agent = KubernetesAssistantAgent(config=mock_config, session_id=session_id)

        # Use the agent with context manager
        with agent:
            agent.run("test input")

        # Verify the lambda passed to MCPClient
        client_factory = mock_mcp_client_class.call_args[0][0]
        client_factory()

        # Verify stdio_client was called with correct parameters
        assert mock_stdio_client.called
        call_args = mock_stdio_client.call_args[0][0]

        assert call_args.command == "python"
        assert call_args.args == [
            "-m",
            "kubernetes_mcp_server",
            "--read-only",
            "--kubeconfig",
            "/test/path/kubeconfig.yaml",
        ]


class TestKubernetesAssistantAgentEdgeCases:
    """Test edge cases and error scenarios."""

    @patch("kubernetes_assistant.kubernetes_assistant_agent.FileSessionManager")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.Agent")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.MCPClient")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.stdio_client")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.SummarizingConversationManager")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.agent_prompt")
    def test_run_with_empty_input(
        self,
        mock_agent_prompt,
        mock_conv_manager,
        mock_stdio_client,
        mock_mcp_client_class,
        mock_agent_class,
        mock_session_manager,
        mock_config,
        mock_model,
        session_id,
    ):
        """Test that run handles empty input string."""
        # Setup mocks
        mock_mcp_instance = MagicMock()
        mock_mcp_client_class.return_value = mock_mcp_instance
        mock_mcp_instance.list_tools_sync.return_value = []

        mock_agent_instance = MagicMock()
        mock_agent_result = Mock()
        mock_agent_result.output = ""
        mock_agent_instance.return_value = mock_agent_result
        mock_agent_class.return_value = mock_agent_instance

        # Create agent with mocked dependencies
        with patch("kubernetes_assistant.config.ModelConfig.create_model", return_value=mock_model):
            agent = KubernetesAssistantAgent(config=mock_config, session_id=session_id)

        # Enter context manager and run with empty input
        with agent:
            _result = agent.run("")

        # Verify it still calls the agent
        mock_agent_instance.assert_called_once_with("")

    @patch("kubernetes_assistant.kubernetes_assistant_agent.FileSessionManager")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.Agent")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.MCPClient")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.stdio_client")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.SummarizingConversationManager")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.agent_prompt")
    def test_run_with_no_tools_available(
        self,
        mock_agent_prompt,
        mock_conv_manager,
        mock_stdio_client,
        mock_mcp_client_class,
        mock_agent_class,
        mock_session_manager,
        mock_config,
        mock_model,
        session_id,
    ):
        """Test that agent handles the case when no tools are available."""
        # Setup mocks
        mock_mcp_instance = MagicMock()
        mock_mcp_client_class.return_value = mock_mcp_instance
        mock_mcp_instance.list_tools_sync.return_value = []  # No tools

        mock_agent_instance = MagicMock()
        mock_agent_result = Mock()
        mock_agent_result.output = "no tools available"
        mock_agent_instance.return_value = mock_agent_result
        mock_agent_class.return_value = mock_agent_instance

        # Create agent with mocked dependencies
        with patch("kubernetes_assistant.config.ModelConfig.create_model", return_value=mock_model):
            agent = KubernetesAssistantAgent(config=mock_config, session_id=session_id)

        # Enter context manager and run
        with agent:
            result = agent.run("test input")

        # Verify Agent was created with empty tools list
        assert mock_agent_class.call_args[1]["tools"] == []
        assert result == mock_agent_result

    @patch("kubernetes_assistant.kubernetes_assistant_agent.FileSessionManager")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.Agent")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.MCPClient")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.stdio_client")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.SummarizingConversationManager")
    @patch("kubernetes_assistant.kubernetes_assistant_agent.agent_prompt")
    def test_run_with_long_input(
        self,
        mock_agent_prompt,
        mock_conv_manager,
        mock_stdio_client,
        mock_mcp_client_class,
        mock_agent_class,
        mock_session_manager,
        mock_config,
        mock_model,
        session_id,
    ):
        """Test that run handles long input strings."""
        # Setup mocks
        mock_mcp_instance = MagicMock()
        mock_mcp_client_class.return_value = mock_mcp_instance
        mock_mcp_instance.list_tools_sync.return_value = []

        mock_agent_instance = MagicMock()
        mock_agent_result = Mock()
        mock_agent_result.output = "processed long input"
        mock_agent_instance.return_value = mock_agent_result
        mock_agent_class.return_value = mock_agent_instance

        # Create a long input string
        long_input = "What is the status? " * 500

        # Create agent with mocked dependencies
        with patch("kubernetes_assistant.config.ModelConfig.create_model", return_value=mock_model):
            agent = KubernetesAssistantAgent(config=mock_config, session_id=session_id)

        # Enter context manager and run with long input
        with agent:
            _result = agent.run(long_input)

        # Verify the agent was called with the full input
        mock_agent_instance.assert_called_once_with(long_input)
