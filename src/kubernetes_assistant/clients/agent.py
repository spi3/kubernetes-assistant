import contextlib

from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.agent import AgentResult
from strands.agent.conversation_manager import SummarizingConversationManager
from strands.models.model import Model
from strands.session.file_session_manager import FileSessionManager
from strands.tools.mcp import MCPClient

from kubernetes_assistant.config import KubernetesAssistantConfig
from kubernetes_assistant.prompts.agent_prompt import agent_prompt


class KubernetesAssistantAgent:
    def __init__(self, config: KubernetesAssistantConfig, model: Model, session_id: str):
        self.config = config
        self.model = model
        self.session_id = session_id

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

        # Enter the kubernetes client context
        self._context_stack.enter_context(self.kubernetes_mcp_client)

        # Enter the prometheus client context if it exists
        if self.prometheus_mcp_client is not None:
            self._context_stack.enter_context(self.prometheus_mcp_client)

        # Now that contexts are entered, we can list tools
        tools = self.kubernetes_mcp_client.list_tools_sync()
        if self.prometheus_mcp_client is not None:
            tools += self.prometheus_mcp_client.list_tools_sync()

        # Initialize the agent with tools
        self.agent = Agent(
            model=self.model,
            tools=tools,
            system_prompt=agent_prompt(
                self.config.agent_name, self.config.cluster_name, self.config.agent_role
            ),
            conversation_manager=SummarizingConversationManager(),
            session_manager=FileSessionManager(
                session_id=self.session_id, storage_dir=self.config.config_dir + "/sessions"
            ),
        )

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

        agent_result = self.agent(input)  # Prints model output to stdout by default
        return agent_result
