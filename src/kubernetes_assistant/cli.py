from strands.models.ollama import OllamaModel

from kubernetes_assistant.clients.agent import KubernetesAssistantAgent
from kubernetes_assistant.config import KubernetesAssistantConfig


def cli() -> None:
    kube_assistant_config = KubernetesAssistantConfig()

    model = OllamaModel(
        host=kube_assistant_config.llm_config.model_host,  # Ollama server address
        model_id=kube_assistant_config.llm_config.model_id,  # Specify which model to use
    )

    with KubernetesAssistantAgent(kube_assistant_config, model, "cli-session") as agent:
        while True:
            query = input("Enter your Kubernetes query (or 'exit' to quit): ")
            agent.run(query)
            print()


if __name__ == "__main__":
    cli()
