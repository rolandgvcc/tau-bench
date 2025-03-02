import abc
from typing import Any, TypeVar

from pydantic import BaseModel

from tau_bench.model_utils.api.datapoint import (
    BinaryClassifyDatapoint,
    ClassifyDatapoint,
    GenerateDatapoint,
    ParseDatapoint,
    ParseForceDatapoint,
    ScoreDatapoint,
)
from tau_bench.model_utils.api.types import PartialObj
from tau_bench.model_utils.model.model import (
    BinaryClassifyModel,
    ClassifyModel,
    GenerateModel,
    ParseForceModel,
    ParseModel,
    Platform,
    ScoreModel,
)

T = TypeVar("T", bound=BaseModel)

LLM_SAMPLING_TEMPERATURE_EPS = 1e-5


def wrap_temperature(temperature: float) -> float:
    return max(temperature, LLM_SAMPLING_TEMPERATURE_EPS)


class GeneralModel(
    ClassifyModel,
    BinaryClassifyModel,
    ParseModel,
    GenerateModel,
    ParseForceModel,
    ScoreModel,
):
    @abc.abstractmethod
    def classify(
        self,
        instruction: str,
        text: str,
        options: list[str],
        examples: list[ClassifyDatapoint] | None = None,
        temperature: float | None = None,
    ) -> int:
        raise NotImplementedError

    def binary_classify(
        self,
        instruction: str,
        text: str,
        examples: list[BinaryClassifyDatapoint] | None = None,
        temperature: float | None = None,
    ) -> bool:
        return (
            self.classify(
                instruction,
                text,
                ["true", "false"],
                examples=(
                    None
                    if examples is None
                    else [
                        ClassifyDatapoint(
                            instruction=example.instruction,
                            text=example.text,
                            options=["true", "false"],
                            response=0 if example.response else 1,
                        )
                        for example in examples
                    ]
                ),
                temperature=temperature,
            )
            == 0
        )

    @abc.abstractmethod
    def parse(
        self,
        text: str,
        typ: type[T] | dict[str, Any],
        examples: list[ParseDatapoint] | None = None,
        temperature: float | None = None,
    ) -> T | PartialObj | dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def generate(
        self,
        instruction: str,
        text: str,
        examples: list[GenerateDatapoint] | None = None,
        temperature: float | None = None,
    ) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def parse_force(
        self,
        instruction: str,
        typ: type[T] | dict[str, Any],
        text: str | None = None,
        examples: list[ParseForceDatapoint] | None = None,
        temperature: float | None = None,
    ) -> T | dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def score(
        self,
        instruction: str,
        text: str,
        min: int,
        max: int,
        examples: list[ScoreDatapoint] | None = None,
        temperature: float | None = None,
    ) -> int:
        raise NotImplementedError


def model_factory(
    model_id: str,
    platform: str | Platform,
    api_key: str | None = None,
    temperature: float = 0.0,
) -> GeneralModel:
    if isinstance(platform, str):
        platform = Platform(platform)
    if platform == Platform.OPENAI:
        from tau_bench.model_utils.model.openai import OpenAIModel

        return OpenAIModel(model=model_id, api_key=api_key, temperature=temperature)

    else:
        raise ValueError("base_url must be provided for custom models")
