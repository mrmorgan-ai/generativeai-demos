import asyncio
from autogen_core.models import UserMessage
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

async def main():
    anthropic_client = AnthropicChatCompletionClient(model="claude-3-7-sonnet-20250219")
    result = await anthropic_client.create(
        [UserMessage(content="What is the capital of France?", source="user")]
    )
    print(result)

# Run the async function
asyncio.run(main())
