"""Discord integration for kubernetes-assistant."""

import asyncio
import logging

import discord

logger = logging.getLogger(__name__)


class DiscordClient:
    """Discord client for sending and receiving messages."""

    def __init__(self, token: str) -> None:
        """
        Initialize the Discord client.

        Args:
            token: Discord bot token
        """
        self.token = token
        # Set up intents for message content
        intents = discord.Intents.default()
        intents.message_content = True
        self.client = discord.Client(intents=intents)
        self._ready_event = asyncio.Event()
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Set up Discord event handlers."""

        @self.client.event
        async def on_ready() -> None:
            """Handle bot ready event."""
            logger.info(f"Logged in as {self.client.user}")
            self._ready_event.set()

        @self.client.event
        async def on_message(message: discord.Message) -> None:
            """Handle incoming messages."""
            # Don't process messages from the bot itself
            if message.author == self.client.user:
                return

            # Only process messages where the bot is mentioned
            if self.client.user not in message.mentions:
                return

            await self._message_queue.put(message)

    async def start(self) -> None:
        """Start the Discord client."""
        asyncio.create_task(self.client.start(self.token))
        await self._ready_event.wait()

    async def close(self) -> None:
        """Close the Discord client connection."""
        await self.client.close()

    async def wait_for_message(
        self,
        timeout: float | None = None,
    ) -> discord.Message:
        """
        Block until a message is received in the specified channels.

        Args:
            timeout: Optional timeout in seconds. If None, waits indefinitely.

        Returns:
            The received Discord message.

        Raises:
            asyncio.TimeoutError: If timeout is reached before a message is received.
        """

        # Clear any existing messages in the queue
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Wait for a message with optional timeout
        if timeout:
            try:
                message = await asyncio.wait_for(self._message_queue.get(), timeout=timeout)
            except TimeoutError:
                raise
        else:
            message = await self._message_queue.get()

        return message

    async def send_message(
        self,
        channel_id: int,
        content: str,
        embed: discord.Embed | None = None,
    ) -> discord.Message:
        """
        Send a message to the specified channel.

        If the content exceeds 2000 characters, it will be split into multiple messages.

        Args:
            channel_id: The ID of the channel to send the message to.
            content: The text content of the message.
            embed: Optional embed to include with the message.

        Returns:
            The last sent Discord message.

        Raises:
            discord.NotFound: If the channel doesn't exist or bot doesn't have access.
            discord.Forbidden: If the bot doesn't have permission to send messages in the channel.
            discord.HTTPException: If sending the message fails.
        """
        channel = self.client.get_channel(channel_id)

        if channel is None:
            # Try fetching the channel if it's not in cache
            channel = await self.client.fetch_channel(channel_id)

        if not isinstance(channel, discord.TextChannel):
            raise ValueError(f"Channel {channel_id} is not a text channel")

        # Discord's max message length is 2000 characters
        MAX_LENGTH = 2000

        # If content fits in one message, send it normally
        if len(content) <= MAX_LENGTH:
            if embed:
                message = await channel.send(content=content, embed=embed)
            else:
                message = await channel.send(content=content)
            return message

        # Split content into chunks
        chunks = self._split_message(content, MAX_LENGTH)

        # Send first chunk with embed if provided
        if embed:
            message = await channel.send(content=chunks[0], embed=embed)
        else:
            message = await channel.send(content=chunks[0])

        # Send remaining chunks
        for chunk in chunks[1:]:
            message = await channel.send(content=chunk)

        return message

    @staticmethod
    def _split_message(content: str, max_length: int) -> list[str]:
        """
        Split a message into chunks that fit within Discord's character limit.

        Tries to split at newlines or spaces to avoid breaking words.

        Args:
            content: The message content to split.
            max_length: The maximum length for each chunk.

        Returns:
            A list of message chunks.
        """
        if len(content) <= max_length:
            return [content]

        chunks = []
        remaining = content

        while remaining:
            if len(remaining) <= max_length:
                chunks.append(remaining)
                break

            # Try to find a good split point
            split_point = max_length

            # First, try to split at a newline
            last_newline = remaining[:max_length].rfind("\n")
            if last_newline > max_length * 0.5:  # Only use if it's not too early
                split_point = last_newline + 1

            # If no good newline, try to split at a space
            elif " " in remaining[:max_length]:
                last_space = remaining[:max_length].rfind(" ")
                if last_space > max_length * 0.5:  # Only use if it's not too early
                    split_point = last_space + 1

            # Add the chunk and continue with the rest
            chunks.append(remaining[:split_point])
            remaining = remaining[split_point:]

        return chunks

    async def __aenter__(self) -> "DiscordClient":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
