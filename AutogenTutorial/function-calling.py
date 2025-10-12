import asyncio
from typing import Literal

from autogen_core.models import (
    AssistantMessage,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel


# Define the structured output format.
class AgentResponse(BaseModel):
    thoughts: str
    response: Literal["happy", "sad", "neutral"]


# Define the function to be called as a tool.
def sentiment_analysis(text: str) -> str:
    """Given a text, return the sentiment."""
    return "happy" if "happy" in text else "sad" if "sad" in text else "neutral"


# Create a FunctionTool instance with `strict=True`,
# which is required for structured output mode.
tool = FunctionTool(sentiment_analysis, description="Sentiment Analysis", strict=True)

# Create an OpenAIChatCompletionClient instance.
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    response_format=AgentResponse,  # type: ignore
)


async def main() -> None:
    # Generate a response using the tool.
    response1 = await model_client.create(
        messages=[
            SystemMessage(
                content="Analyze input text sentiment using the tool provided."
            ),
            UserMessage(content="I am happy.", source="user"),
        ],
        tools=[tool],
    )
    print(response1.content)
    # Should be a list of tool calls.
    # [FunctionCall(name="sentiment_analysis", arguments={"text": "I am happy."}, ...)]

    assert isinstance(response1.content, list)
    response2 = await model_client.create(
        messages=[
            SystemMessage(
                content="Analyze input text sentiment using the tool provided."
            ),
            UserMessage(content="I am happy.", source="user"),
            AssistantMessage(content=response1.content, source="assistant"),
            FunctionExecutionResultMessage(
                content=[
                    FunctionExecutionResult(
                        content="happy",
                        call_id=response1.content[0].id,
                        is_error=False,
                        name="sentiment_analysis",
                    )
                ]
            ),
        ],
    )
    print(response2.content)
    # Should be a structured output.
    # {"thoughts": "The user is happy.", "response": "happy"}


asyncio.run(main())
