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
        self.session_manager = FileSessionManager(session_id=session_id, storage_dir=self.config.config_dir + "/sessions")

    def run(self, input: str) -> AgentResult:
        stdio_mcp_client = MCPClient(
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

        with stdio_mcp_client:
            tools = stdio_mcp_client.list_tools_sync()

            # Create an agent using the Ollama model
            agent = Agent(
                model=self.model,
                tools=tools,
                system_prompt=agent_prompt(
                    self.config.agent_name, self.config.cluster_name, self.config.agent_role
                ),
                conversation_manager=SummarizingConversationManager(),
                session_manager=self.session_manager,
            )

            # Use the agent
            agent_result = agent(input)  # Prints model output to stdout by default
            return agent_result
