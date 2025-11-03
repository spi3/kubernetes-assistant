import contextlib
import logging

from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.agent import AgentResult
from strands.agent.conversation_manager import SummarizingConversationManager
from strands.session.file_session_manager import FileSessionManager
from strands.tools.mcp import MCPClient

from kubernetes_assistant.kubernetes_assistant_config import KubernetesAssistantConfig
from kubernetes_assistant.prompts.agent_prompt import agent_prompt

logger = logging.getLogger(__name__)


class KubernetesAssistantAgent:
    def __init__(self, config: KubernetesAssistantConfig, session_id: str):
        self.config = config
        self.session_id = session_id

        try:
            self.model = config.llm_config.create_model()
        except Exception as e:
            logger.error(f"Failed to create LLM model: {e}")
            raise RuntimeError(f"Failed to initialize LLM model: {e}") from e

        # MCP Clients
        self.kubernetes_mcp_client = MCPClient(
            lambda: stdio_client(
                StdioServerParameters(
                    command="python",
                    args=[
                        "-m",
                        "kubernetes_mcp_server",
                        "--read-only",
                        "--kubeconfig",
                        f"{self.config.kubeconfig_path}",
                    ],
                )
            )
        )

        self.prometheus_mcp_client = None
        if self.config.prometheus_url is not None:
            prometheus_url: str = self.config.prometheus_url
            self.prometheus_mcp_client = MCPClient(
                lambda: stdio_client(
                    StdioServerParameters(
                        command="docker",
                        args=[
                            "run",
                            "-i",
                            "--rm",
                            "-e",
                            "PROMETHEUS_URL",
                            "ghcr.io/pab1it0/prometheus-mcp-server",
                        ],
                        env={"PROMETHEUS_URL": prometheus_url},
                    )
                )
            )

        self.agent = None
        self._context_stack = None

    def __enter__(self):
        """Enter the context manager and start MCP clients."""
        self._context_stack = contextlib.ExitStack()
        self._context_stack.__enter__()

        try:
            # Enter the kubernetes client context
            try:
                self._context_stack.enter_context(self.kubernetes_mcp_client)
                logger.info("Kubernetes MCP client connected successfully")
            except Exception as e:
                logger.error(f"Failed to connect to Kubernetes MCP client: {e}")
                raise RuntimeError(f"Failed to connect to Kubernetes MCP server: {e}") from e

            # Enter the prometheus client context if it exists
            if self.prometheus_mcp_client is not None:
                try:
                    self._context_stack.enter_context(self.prometheus_mcp_client)
                    logger.info("Prometheus MCP client connected successfully")
                except Exception as e:
                    logger.warning(
                        f"Failed to connect to Prometheus MCP client: {e}. "
                        "Continuing without Prometheus support."
                    )
                    self.prometheus_mcp_client = None

            # Now that contexts are entered, we can list tools
            try:
                tools = self.kubernetes_mcp_client.list_tools_sync()
                logger.info(f"Loaded {len(tools)} tools from Kubernetes MCP server")
            except Exception as e:
                logger.error(f"Failed to list tools from Kubernetes MCP client: {e}")
                raise RuntimeError(
                    f"Failed to retrieve tools from Kubernetes MCP server: {e}"
                ) from e

            if self.prometheus_mcp_client is not None:
                try:
                    prometheus_tools = self.prometheus_mcp_client.list_tools_sync()
                    tools += prometheus_tools
                    logger.info(f"Loaded {len(prometheus_tools)} tools from Prometheus MCP server")
                except Exception as e:
                    logger.warning(
                        f"Failed to list tools from Prometheus MCP client: {e}. "
                        "Continuing without Prometheus tools."
                    )

            # Initialize the agent with tools
            try:
                # Use custom prompt if provided, otherwise generate the default prompt
                system_prompt = (
                    self.config.custom_agent_prompt
                    if self.config.custom_agent_prompt is not None
                    else agent_prompt(
                        self.config.agent_name,
                        self.config.cluster_name,
                        self.config.agent_role,
                    )
                )

                self.agent = Agent(
                    model=self.model,
                    tools=tools,
                    system_prompt=system_prompt,
                    conversation_manager=SummarizingConversationManager(),
                    session_manager=FileSessionManager(
                        session_id=self.session_id,
                        storage_dir=self.config.config_dir + "/sessions",
                    ),
                )
                logger.info("Agent initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize agent: {e}")
                raise RuntimeError(f"Failed to initialize agent: {e}") from e

        except Exception:
            # Clean up on error
            if self._context_stack is not None:
                self._context_stack.__exit__(None, None, None)
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and clean up MCP clients."""
        if self._context_stack is not None:
            self._context_stack.__exit__(exc_type, exc_val, exc_tb)
        return False

    def run(self, input: str) -> AgentResult:
        if self.agent is None:
            raise RuntimeError(
                "Agent has not been initialized. Use the context manager to initialize."
            )

        try:
            agent_result = self.agent(input)  # Prints model output to stdout by default
            return agent_result
        except Exception as e:
            logger.error(f"Agent execution failed for input: {input[:100]}... Error: {e}")
            raise RuntimeError(f"Agent execution failed: {e}") from e
