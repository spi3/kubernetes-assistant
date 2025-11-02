import asyncio
import logging
import os

from strands.models.ollama import OllamaModel

from kubernetes_assistant.clients.agent import KubernetesAssistantAgent
from kubernetes_assistant.clients.discord import DiscordClient
from kubernetes_assistant.config import KubernetesAssistantConfig
from kubernetes_assistant.utils.discord_message_formatter import discord_message_formatter


async def main_async(kube_assistant_config: KubernetesAssistantConfig) -> None:
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    logger = logging.getLogger(__name__)

    if kube_assistant_config.discord_token is None:
        raise ValueError("Discord token is required.")

    model = OllamaModel(
        host=kube_assistant_config.llm_config.model_host,  # Ollama server address
        model_id=kube_assistant_config.llm_config.model_id,  # Specify which model to use
    )

    async with DiscordClient(kube_assistant_config.discord_token) as client:
        while True:
            # Block until message received in specific channels
            message = await client.wait_for_message()
            formatted_message = discord_message_formatter(message)
            logger.info(f"Received discord message: {formatted_message}")

            guild_id = message.guild.id if message.guild else "dm"
            with KubernetesAssistantAgent(
                kube_assistant_config, model, f"{guild_id}-{message.channel.id}"
            ) as agent:
                result = agent.run(formatted_message)
                # Handle ContentBlock properly - it might be a TextBlock or other type
                content_block = result.message["content"][0]
                result_content = content_block.get("text", str(content_block))
                logger.info(f"Agent Response to discord message: {result_content}")

                await client.send_message(
                    channel_id=message.channel.id,
                    content=result_content,
                )

                logger.info("Sent response back to discord.")


def main() -> None:
    config = KubernetesAssistantConfig()  # type: ignore[call-arg]
    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
