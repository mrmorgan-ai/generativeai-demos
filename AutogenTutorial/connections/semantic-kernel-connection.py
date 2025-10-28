import os
import asyncio


from autogen_core.models import UserMessage
from autogen_ext.models.semantic_kernel import SKChatCompletionAdapter  # type: ignore
from semantic_kernel import Kernel # type: ignore
from semantic_kernel.connectors.ai.anthropic import (
    AnthropicChatCompletion, # type: ignore
    AnthropicChatPromptExecutionSettings # type: ignore
) #typed: ignore

from semantic_kernel.memory.null_memory import NullMemory # type: ignore

sk_client = AnthropicChatCompletion(
    ai_model_id="claude-3-5-sonnet-20241022",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    service_id="my-service-id",  # Optional; for targeting specific services within Semantic Kernel
)

settings = AnthropicChatPromptExecutionSettings(
    temperature=0.2,
)

anthropic_model_client = SKChatCompletionAdapter(
    sk_client, kernel=Kernel(memory=NullMemory()), prompt_settings=settings
)

# Call the model directly.
async def main():
    model_result = await anthropic_model_client.create(  # noqa: F704
        messages=[UserMessage(content="What is the capital of France?", source="User")]
    )
    print(model_result)

if __name__ == "__main__":
    asyncio.run(main())
