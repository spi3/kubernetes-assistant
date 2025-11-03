import logging
import sys

from kubernetes_assistant.kubernetes_assistant_agent import KubernetesAssistantAgent
from kubernetes_assistant.kubernetes_assistant_config import KubernetesAssistantConfig

logger = logging.getLogger(__name__)


def cli() -> None:
    """Run the interactive CLI for Kubernetes Assistant."""
    logging.basicConfig(level=logging.INFO)

    try:
        kube_assistant_config = KubernetesAssistantConfig()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        print("Error: Failed to load configuration. Please check your environment variables.")
        sys.exit(1)

    try:
        with KubernetesAssistantAgent(kube_assistant_config, "cli-session") as agent:
            print("Kubernetes Assistant CLI")
            print("Type 'exit' or press Ctrl+C to quit")
            print()

            while True:
                try:
                    query = input("Enter your Kubernetes query (or 'exit' to quit): ")

                    # Check for exit command
                    if query.lower().strip() in ["exit", "quit", "q"]:
                        print("Goodbye!")
                        break

                    # Skip empty queries
                    if not query.strip():
                        continue

                    try:
                        result = agent.run(query)
                        # Extract and display the result
                        try:
                            content_block = result.message["content"][0]
                            result_content = content_block.get("text", str(content_block))
                            print(f"\n{result_content}\n")
                        except (KeyError, IndexError, AttributeError) as e:
                            logger.error(f"Failed to extract agent result: {e}")
                            print("\nError: Failed to process agent response. Please try again.\n")
                    except Exception as e:
                        logger.error(f"Agent execution failed: {e}", exc_info=True)
                        print(
                            f"\nError: Failed to process your query: {e}\n"
                            "Please try again or check your cluster configuration.\n"
                        )

                except EOFError:
                    # Handle Ctrl+D
                    print("\nGoodbye!")
                    break
                except KeyboardInterrupt:
                    # Handle Ctrl+C
                    print("\n\nGoodbye!")
                    break

    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}", exc_info=True)
        print(f"Error: Failed to initialize Kubernetes Assistant: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
