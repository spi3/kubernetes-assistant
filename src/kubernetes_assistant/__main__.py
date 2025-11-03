import asyncio
import logging
import os

from kubernetes_assistant.clients.discord import DiscordClient
from kubernetes_assistant.config import KubernetesAssistantConfig
from kubernetes_assistant.kubernetes_assistant_agent import KubernetesAssistantAgent
from kubernetes_assistant.utils.discord_message_formatter import discord_message_formatter


async def main_async(kube_assistant_config: KubernetesAssistantConfig) -> None:
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    logger = logging.getLogger(__name__)

    if kube_assistant_config.discord_token is None:
        raise ValueError("Discord token is required.")

    async with DiscordClient(kube_assistant_config.discord_token) as client:
        while True:
            try:
                # Block until message received in specific channels
                message = await client.wait_for_message()
                formatted_message = discord_message_formatter(message)
                logger.info(f"Received discord message: {formatted_message}")

                # Safely extract guild and channel IDs
                try:
                    guild_id = message.guild.id if message.guild else "dm"
                    channel_id = message.channel.id
                except AttributeError as e:
                    logger.error(f"Failed to extract message attributes: {e}")
                    continue

                try:
                    with KubernetesAssistantAgent(
                        kube_assistant_config, f"{guild_id}-{channel_id}"
                    ) as agent:
                        result = agent.run(formatted_message)

                        # Handle ContentBlock properly - it might be a TextBlock or other type
                        try:
                            content_block = result.message["content"][0]
                            result_content = content_block.get("text", str(content_block))
                        except (KeyError, IndexError, AttributeError) as e:
                            logger.error(f"Failed to extract agent result content: {e}")
                            result_content = (
                                "Sorry, I encountered an error processing the response."
                            )

                        logger.info(f"Agent Response to discord message: {result_content}")

                        try:
                            await client.send_message(
                                channel_id=channel_id,
                                content=result_content,
                            )
                            logger.info("Sent response back to discord.")
                        except Exception as e:
                            logger.error(f"Failed to send message to Discord: {e}")

                except Exception as e:
                    logger.error(f"Agent execution failed: {e}", exc_info=True)
                    try:
                        await client.send_message(
                            channel_id=channel_id,
                            content="Sorry, I encountered an error processing your request. "
                            "Please try again later.",
                        )
                    except Exception as send_error:
                        logger.error(f"Failed to send error message to Discord: {send_error}")

            except asyncio.CancelledError:
                logger.info("Bot shutting down...")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in message loop: {e}", exc_info=True)
                # Continue processing next message
                continue


def main() -> None:
    try:
        config = KubernetesAssistantConfig()  # type: ignore[call-arg]
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}", exc_info=True)
        raise SystemExit(1) from e

    try:
        asyncio.run(main_async(config))
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
