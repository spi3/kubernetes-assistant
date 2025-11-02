import discord


def discord_message_formatter(message: discord.Message) -> str:
    """Format the agent result into a Discord-compatible message string."""
    embeds = message.embeds
    timestamp = message.created_at.__str__()
    content = message.content
    author = message.author.display_name

    embed_parts = []
    for embed in embeds:
        parts = []
        if embed.title:
            parts.append(embed.title)
        if embed.description:
            parts.append(embed.description)
        if embed.url:
            parts.append(embed.url)
        if parts:
            embed_parts.append("\n".join(parts))

    embeds_str = "\n".join(embed_parts)

    return f"""
[Message from {author} at {timestamp}]:
{embeds_str}
{content}
"""
