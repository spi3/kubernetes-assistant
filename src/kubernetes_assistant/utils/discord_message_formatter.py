import discord


def discord_message_formatter(message: discord.Message) -> str:
    """Format the agent result into a Discord-compatible message string."""
    embeds = message.embeds
    timestamp = message.created_at.__str__()
    content = message.content
    author = message.author.display_name

    return f"""
[Message from {author} at {timestamp}]:
{embeds}
{content}
"""
