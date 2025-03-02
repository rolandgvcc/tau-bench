import os

from tau_bench.model_utils.api.datapoint import Datapoint
from tau_bench.model_utils.model.chat import ChatModel, Message
from tau_bench.model_utils.model.completion import (
    approx_prompt_str,
)
from tau_bench.model_utils.model.general_model import wrap_temperature
from tau_bench.model_utils.model.utils import approx_num_tokens
from tau_bench.constants import DEFAULT_MODEL, MAX_CONTEXT_LENGTH, DEFAULT_BASE_URL

from openai import AsyncOpenAI, OpenAI


class OpenAIModel(ChatModel):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:

        if model is None:
            self.model = DEFAULT_MODEL
        else:
            self.model = model

        api_key = None
        if api_key is None:
            api_key = os.getenv("XAI_API_KEY")
            if api_key is None:
                raise ValueError("XAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.temperature = temperature

    def generate_message(
        self,
        messages: list[Message],
        force_json: bool,
        temperature: float | None = None,
    ) -> Message:
        if temperature is None:
            temperature = self.temperature
        msgs = self.build_generate_message_state(messages)
        res = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=wrap_temperature(temperature),
            response_format={"type": "json_object" if force_json else "text"},
        )
        return self.handle_generate_message_response(
            prompt=msgs, content=res.choices[0].message.content, force_json=force_json
        )

    def supports_dp(self, dp: Datapoint) -> bool:
        prompt = approx_prompt_str(dp)
        return approx_num_tokens(prompt) <= MAX_CONTEXT_LENGTH
