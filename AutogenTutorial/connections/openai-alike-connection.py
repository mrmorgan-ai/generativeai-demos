from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import UserMessage


from autogen_core.models import ModelFamily

openai_client = OpenAIChatCompletionClient(
    model="gpt-4o-2024-08-06",
    # api_key="sk-...", # Optional if you have an OPENAI_API_KEY environment variable set.
)

result = await openai_client.create(  # noqa: F704
    [UserMessage(content="What is the capital of France?", source="user")]
)  # type: ignore

print(result)


custom_model_client = OpenAIChatCompletionClient(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434/v1",
    api_key="placeholder",
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": ModelFamily.R1,
    },
)
