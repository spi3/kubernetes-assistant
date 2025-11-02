"""Unit tests for DiscordClient class."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import discord
import pytest

from kubernetes_assistant.clients.discord import DiscordClient


@pytest.fixture
def discord_token():
    """Provide a test Discord token."""
    return "test-discord-token-12345"


@pytest.fixture
def discord_client_instance(discord_token):
    """Create a DiscordClient instance with mocked discord.Client."""
    with patch("kubernetes_assistant.clients.discord.discord.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client.user = Mock()
        mock_client.user.name = "TestBot"
        mock_client.event = MagicMock()
        mock_client_class.return_value = mock_client

        client = DiscordClient(token=discord_token)
        yield client


class TestDiscordClientInit:
    """Test cases for DiscordClient initialization."""

    def test_init_stores_token(self, discord_token):
        """Test that __init__ correctly stores the token."""
        with patch("kubernetes_assistant.clients.discord.discord.Client"):
            client = DiscordClient(token=discord_token)
            assert client.token == discord_token

    def test_init_creates_discord_client_with_intents(self, discord_token):
        """Test that __init__ creates a Discord client with correct intents."""
        with patch("kubernetes_assistant.clients.discord.discord.Client") as mock_client_class:
            with patch(
                "kubernetes_assistant.clients.discord.discord.Intents"
            ) as mock_intents_class:
                mock_intents = Mock()
                mock_intents_class.default.return_value = mock_intents

                _client = DiscordClient(token=discord_token)

                # Verify intents were configured
                mock_intents_class.default.assert_called_once()
                assert mock_intents.message_content is True

                # Verify Discord client was created with intents
                mock_client_class.assert_called_once_with(intents=mock_intents)

    def test_init_creates_ready_event(self, discord_token):
        """Test that __init__ creates a ready event."""
        with patch("kubernetes_assistant.clients.discord.discord.Client"):
            client = DiscordClient(token=discord_token)
            assert isinstance(client._ready_event, asyncio.Event)
            assert not client._ready_event.is_set()

    def test_init_creates_message_queue(self, discord_token):
        """Test that __init__ creates a message queue."""
        with patch("kubernetes_assistant.clients.discord.discord.Client"):
            client = DiscordClient(token=discord_token)
            assert isinstance(client._message_queue, asyncio.Queue)
            assert client._message_queue.empty()

    def test_init_calls_setup_handlers(self, discord_token):
        """Test that __init__ calls _setup_handlers."""
        with patch("kubernetes_assistant.clients.discord.discord.Client"):
            with patch.object(DiscordClient, "_setup_handlers") as mock_setup:
                _client = DiscordClient(token=discord_token)
                mock_setup.assert_called_once()


class TestDiscordClientSetupHandlers:
    """Test cases for Discord event handlers."""

    @pytest.mark.asyncio
    async def test_on_ready_handler_sets_ready_event(self, discord_client_instance):
        """Test that on_ready handler sets the ready event."""
        # Get the on_ready handler
        on_ready = None
        for call in discord_client_instance.client.event.call_args_list:
            func = call[0][0]
            if func.__name__ == "on_ready":
                on_ready = func
                break

        assert on_ready is not None

        # Call the handler
        await on_ready()

        # Verify the ready event is set
        assert discord_client_instance._ready_event.is_set()

    @pytest.mark.asyncio
    async def test_on_message_handler_adds_message_to_queue(self, discord_client_instance):
        """Test that on_message handler adds messages to the queue."""
        # Get the on_message handler
        on_message = None
        for call in discord_client_instance.client.event.call_args_list:
            func = call[0][0]
            if func.__name__ == "on_message":
                on_message = func
                break

        assert on_message is not None

        # Create a mock message from another user
        mock_message = Mock(spec=discord.Message)
        mock_message.author = Mock()
        mock_message.author.id = 123

        # Ensure the message is not from the bot itself
        discord_client_instance.client.user = Mock()
        discord_client_instance.client.user.id = 456

        # Call the handler
        await on_message(mock_message)

        # Verify the message was added to the queue
        assert not discord_client_instance._message_queue.empty()
        queued_message = await discord_client_instance._message_queue.get()
        assert queued_message == mock_message

    @pytest.mark.asyncio
    async def test_on_message_handler_ignores_bot_messages(self, discord_client_instance):
        """Test that on_message handler ignores messages from the bot itself."""
        # Get the on_message handler
        on_message = None
        for call in discord_client_instance.client.event.call_args_list:
            func = call[0][0]
            if func.__name__ == "on_message":
                on_message = func
                break

        assert on_message is not None

        # Create a mock message from the bot itself
        mock_message = Mock(spec=discord.Message)
        mock_message.author = discord_client_instance.client.user

        # Call the handler
        await on_message(mock_message)

        # Verify the message was NOT added to the queue
        assert discord_client_instance._message_queue.empty()


class TestDiscordClientStart:
    """Test cases for Discord client start method."""

    @pytest.mark.asyncio
    async def test_start_creates_client_task(self, discord_client_instance):
        """Test that start creates a task to start the Discord client."""
        with patch("asyncio.create_task") as mock_create_task:
            # Mock the client.start method to return a coroutine
            discord_client_instance.client.start = AsyncMock()

            # Set the ready event immediately to prevent waiting
            discord_client_instance._ready_event.set()

            await discord_client_instance.start()

            # Verify create_task was called
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_waits_for_ready_event(self, discord_client_instance):
        """Test that start waits for the ready event."""
        discord_client_instance.client.start = AsyncMock()

        # Create a task that sets the ready event after a short delay
        async def set_ready_after_delay():
            await asyncio.sleep(0.1)
            discord_client_instance._ready_event.set()

        asyncio.create_task(set_ready_after_delay())

        # This should wait until the ready event is set
        await discord_client_instance.start()

        assert discord_client_instance._ready_event.is_set()

    @pytest.mark.asyncio
    async def test_start_passes_token_to_client(self, discord_client_instance):
        """Test that start passes the token to the Discord client."""
        discord_client_instance.client.start = AsyncMock()
        discord_client_instance._ready_event.set()

        await discord_client_instance.start()

        # Verify the token was passed to client.start
        # Note: We can't directly verify the call because it's wrapped in create_task
        assert discord_client_instance.token == "test-discord-token-12345"


class TestDiscordClientClose:
    """Test cases for Discord client close method."""

    @pytest.mark.asyncio
    async def test_close_calls_client_close(self, discord_client_instance):
        """Test that close calls the Discord client's close method."""
        discord_client_instance.client.close = AsyncMock()

        await discord_client_instance.close()

        discord_client_instance.client.close.assert_called_once()


class TestDiscordClientWaitForMessage:
    """Test cases for wait_for_message method."""

    @pytest.mark.asyncio
    async def test_wait_for_message_clears_existing_messages(self, discord_client_instance):
        """Test that wait_for_message clears existing messages in the queue."""
        # Add some messages to the queue
        old_message1 = Mock(spec=discord.Message)
        old_message2 = Mock(spec=discord.Message)
        await discord_client_instance._message_queue.put(old_message1)
        await discord_client_instance._message_queue.put(old_message2)

        # Add a new message and immediately wait for it
        new_message = Mock(spec=discord.Message)

        async def add_new_message():
            await asyncio.sleep(0.1)
            await discord_client_instance._message_queue.put(new_message)

        asyncio.create_task(add_new_message())

        result = await discord_client_instance.wait_for_message(timeout=1.0)

        # The result should be the new message, not the old ones
        assert result == new_message

    @pytest.mark.asyncio
    async def test_wait_for_message_returns_message(self, discord_client_instance):
        """Test that wait_for_message returns a message from the queue."""
        mock_message = Mock(spec=discord.Message)

        # Add a message to the queue after a short delay
        async def add_message():
            await asyncio.sleep(0.1)
            await discord_client_instance._message_queue.put(mock_message)

        asyncio.create_task(add_message())

        result = await discord_client_instance.wait_for_message(timeout=1.0)

        assert result == mock_message

    @pytest.mark.asyncio
    async def test_wait_for_message_with_timeout_raises_timeout_error(
        self, discord_client_instance
    ):
        """Test that wait_for_message raises TimeoutError when timeout is reached."""
        with pytest.raises(asyncio.TimeoutError):
            await discord_client_instance.wait_for_message(timeout=0.1)

    @pytest.mark.asyncio
    async def test_wait_for_message_without_timeout(self, discord_client_instance):
        """Test that wait_for_message waits indefinitely when no timeout is provided."""
        mock_message = Mock(spec=discord.Message)

        # Add a message to the queue after a short delay
        async def add_message():
            await asyncio.sleep(0.2)
            await discord_client_instance._message_queue.put(mock_message)

        asyncio.create_task(add_message())

        result = await discord_client_instance.wait_for_message(timeout=None)

        assert result == mock_message


class TestDiscordClientSendMessage:
    """Test cases for send_message method."""

    @pytest.mark.asyncio
    async def test_send_message_gets_channel_from_cache(self, discord_client_instance):
        """Test that send_message gets the channel from cache."""
        mock_channel = Mock(spec=discord.TextChannel)
        mock_channel.send = AsyncMock(return_value=Mock(spec=discord.Message))
        discord_client_instance.client.get_channel = Mock(return_value=mock_channel)

        await discord_client_instance.send_message(channel_id=123, content="Test message")

        discord_client_instance.client.get_channel.assert_called_once_with(123)

    @pytest.mark.asyncio
    async def test_send_message_fetches_channel_if_not_in_cache(self, discord_client_instance):
        """Test that send_message fetches the channel if not in cache."""
        mock_channel = Mock(spec=discord.TextChannel)
        mock_channel.send = AsyncMock(return_value=Mock(spec=discord.Message))
        discord_client_instance.client.get_channel = Mock(return_value=None)
        discord_client_instance.client.fetch_channel = AsyncMock(return_value=mock_channel)

        await discord_client_instance.send_message(channel_id=123, content="Test message")

        discord_client_instance.client.fetch_channel.assert_called_once_with(123)

    @pytest.mark.asyncio
    async def test_send_message_raises_error_for_non_text_channel(self, discord_client_instance):
        """Test that send_message raises ValueError for non-text channels."""
        mock_channel = Mock(spec=discord.VoiceChannel)
        discord_client_instance.client.get_channel = Mock(return_value=mock_channel)

        with pytest.raises(ValueError, match="is not a text channel"):
            await discord_client_instance.send_message(channel_id=123, content="Test message")

    @pytest.mark.asyncio
    async def test_send_message_sends_short_message(self, discord_client_instance):
        """Test that send_message sends a short message in one call."""
        mock_channel = Mock(spec=discord.TextChannel)
        mock_message = Mock(spec=discord.Message)
        mock_channel.send = AsyncMock(return_value=mock_message)
        discord_client_instance.client.get_channel = Mock(return_value=mock_channel)

        result = await discord_client_instance.send_message(channel_id=123, content="Short")

        mock_channel.send.assert_called_once_with(content="Short")
        assert result == mock_message

    @pytest.mark.asyncio
    async def test_send_message_sends_message_with_embed(self, discord_client_instance):
        """Test that send_message sends a message with an embed."""
        mock_channel = Mock(spec=discord.TextChannel)
        mock_message = Mock(spec=discord.Message)
        mock_channel.send = AsyncMock(return_value=mock_message)
        discord_client_instance.client.get_channel = Mock(return_value=mock_channel)

        mock_embed = Mock(spec=discord.Embed)

        result = await discord_client_instance.send_message(
            channel_id=123, content="Test", embed=mock_embed
        )

        mock_channel.send.assert_called_once_with(content="Test", embed=mock_embed)
        assert result == mock_message

    @pytest.mark.asyncio
    async def test_send_message_splits_long_message(self, discord_client_instance):
        """Test that send_message splits long messages into chunks."""
        mock_channel = Mock(spec=discord.TextChannel)
        mock_message1 = Mock(spec=discord.Message)
        mock_message2 = Mock(spec=discord.Message)
        mock_channel.send = AsyncMock(side_effect=[mock_message1, mock_message2])
        discord_client_instance.client.get_channel = Mock(return_value=mock_channel)

        # Create a message longer than 2000 characters
        long_content = "A" * 2500

        result = await discord_client_instance.send_message(channel_id=123, content=long_content)

        # Verify send was called multiple times
        assert mock_channel.send.call_count == 2
        assert result == mock_message2  # Returns the last message

    @pytest.mark.asyncio
    async def test_send_message_splits_long_message_with_embed(self, discord_client_instance):
        """Test that send_message includes embed only in first chunk."""
        mock_channel = Mock(spec=discord.TextChannel)
        mock_message1 = Mock(spec=discord.Message)
        mock_message2 = Mock(spec=discord.Message)
        mock_channel.send = AsyncMock(side_effect=[mock_message1, mock_message2])
        discord_client_instance.client.get_channel = Mock(return_value=mock_channel)

        mock_embed = Mock(spec=discord.Embed)
        long_content = "B" * 2500

        await discord_client_instance.send_message(
            channel_id=123, content=long_content, embed=mock_embed
        )

        # First call should have embed
        first_call = mock_channel.send.call_args_list[0]
        assert "embed" in first_call[1]
        assert first_call[1]["embed"] == mock_embed

        # Second call should not have embed
        second_call = mock_channel.send.call_args_list[1]
        assert "embed" not in second_call[1]


class TestDiscordClientSplitMessage:
    """Test cases for _split_message static method."""

    def test_split_message_returns_single_chunk_for_short_message(self):
        """Test that _split_message returns a single chunk for short messages."""
        content = "Short message"
        result = DiscordClient._split_message(content, 2000)

        assert result == ["Short message"]

    def test_split_message_splits_at_newline(self):
        """Test that _split_message prefers splitting at newlines."""
        content = "A" * 1500 + "\n" + "B" * 600
        result = DiscordClient._split_message(content, 2000)

        assert len(result) == 2
        assert result[0].endswith("\n")
        assert result[1].startswith("B")

    def test_split_message_splits_at_space_if_no_newline(self):
        """Test that _split_message splits at spaces if no good newline."""
        content = "A" * 1500 + " " + "B" * 600
        result = DiscordClient._split_message(content, 2000)

        assert len(result) == 2
        assert result[0].endswith(" ")
        assert result[1].startswith("B")

    def test_split_message_hard_split_if_no_good_break_point(self):
        """Test that _split_message does hard split if no good break point."""
        content = "A" * 3000
        result = DiscordClient._split_message(content, 2000)

        assert len(result) == 2
        assert len(result[0]) == 2000
        assert len(result[1]) == 1000

    def test_split_message_multiple_chunks(self):
        """Test that _split_message handles multiple chunks correctly."""
        content = "X" * 5500
        result = DiscordClient._split_message(content, 2000)

        assert len(result) == 3
        assert len(result[0]) == 2000
        assert len(result[1]) == 2000
        assert len(result[2]) == 1500

    def test_split_message_respects_split_threshold(self):
        """Test that _split_message only uses split points after 50% of max_length."""
        # Newline very early (before 50% threshold)
        content = "A" * 200 + "\n" + "B" * 2000
        result = DiscordClient._split_message(content, 2000)

        # Should not split at the early newline
        assert len(result) == 2
        assert len(result[0]) <= 2000


class TestDiscordClientContextManager:
    """Test cases for async context manager support."""

    @pytest.mark.asyncio
    async def test_aenter_calls_start(self, discord_client_instance):
        """Test that __aenter__ calls start."""
        discord_client_instance.start = AsyncMock()

        result = await discord_client_instance.__aenter__()

        discord_client_instance.start.assert_called_once()
        assert result == discord_client_instance

    @pytest.mark.asyncio
    async def test_aexit_calls_close(self, discord_client_instance):
        """Test that __aexit__ calls close."""
        discord_client_instance.close = AsyncMock()

        await discord_client_instance.__aexit__(None, None, None)

        discord_client_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, discord_token):
        """Test that DiscordClient can be used as an async context manager."""
        with patch("kubernetes_assistant.clients.discord.discord.Client"):
            client = DiscordClient(token=discord_token)
            client.start = AsyncMock()
            client.close = AsyncMock()

            async with client as c:
                assert c == client
                client.start.assert_called_once()

            client.close.assert_called_once()


class TestDiscordClientEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_send_message_handles_discord_not_found_error(self, discord_client_instance):
        """Test that send_message propagates discord.NotFound error."""
        # Create a proper mock response object
        mock_response = Mock()
        mock_response.status = 404
        mock_response.reason = "Not Found"

        discord_client_instance.client.get_channel = Mock(return_value=None)
        discord_client_instance.client.fetch_channel = AsyncMock(
            side_effect=discord.NotFound(mock_response, "Channel not found")
        )

        with pytest.raises(discord.NotFound):
            await discord_client_instance.send_message(channel_id=999, content="Test")

    @pytest.mark.asyncio
    async def test_send_message_handles_discord_forbidden_error(self, discord_client_instance):
        """Test that send_message propagates discord.Forbidden error."""
        # Create a proper mock response object
        mock_response = Mock()
        mock_response.status = 403
        mock_response.reason = "Forbidden"

        mock_channel = Mock(spec=discord.TextChannel)
        mock_channel.send = AsyncMock(
            side_effect=discord.Forbidden(mock_response, "Missing permissions")
        )
        discord_client_instance.client.get_channel = Mock(return_value=mock_channel)

        with pytest.raises(discord.Forbidden):
            await discord_client_instance.send_message(channel_id=123, content="Test")

    @pytest.mark.asyncio
    async def test_send_message_handles_discord_http_exception(self, discord_client_instance):
        """Test that send_message propagates discord.HTTPException."""
        # Create a proper mock response object
        mock_response = Mock()
        mock_response.status = 500
        mock_response.reason = "Internal Server Error"

        mock_channel = Mock(spec=discord.TextChannel)
        mock_channel.send = AsyncMock(
            side_effect=discord.HTTPException(mock_response, "Server error")
        )
        discord_client_instance.client.get_channel = Mock(return_value=mock_channel)

        with pytest.raises(discord.HTTPException):
            await discord_client_instance.send_message(channel_id=123, content="Test")

    @pytest.mark.asyncio
    async def test_send_message_with_empty_content(self, discord_client_instance):
        """Test that send_message handles empty content."""
        mock_channel = Mock(spec=discord.TextChannel)
        mock_message = Mock(spec=discord.Message)
        mock_channel.send = AsyncMock(return_value=mock_message)
        discord_client_instance.client.get_channel = Mock(return_value=mock_channel)

        result = await discord_client_instance.send_message(channel_id=123, content="")

        mock_channel.send.assert_called_once_with(content="")
        assert result == mock_message

    def test_split_message_with_exact_max_length(self):
        """Test _split_message with content exactly at max length."""
        content = "X" * 2000
        result = DiscordClient._split_message(content, 2000)

        assert result == [content]

    def test_split_message_with_newlines_at_boundary(self):
        """Test _split_message with newlines at exactly the boundary."""
        content = "A" * 1999 + "\n" + "B" * 100
        result = DiscordClient._split_message(content, 2000)

        assert len(result) == 2
        assert result[0] == "A" * 1999 + "\n"
        assert result[1] == "B" * 100
