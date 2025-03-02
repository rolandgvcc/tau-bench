import abc

from pydantic import BaseModel

from tau_bench.model_utils.api.datapoint import Datapoint, ScoreDatapoint
from tau_bench.model_utils.model.model import Model


class RequestRouter(abc.ABC):
    @abc.abstractmethod
    def route(self, dp: Datapoint, available_models: list[Model]) -> Model:
        raise NotImplementedError


class FirstModelRequestRouter(RequestRouter):
    def route(self, dp: Datapoint, available_models: list[Model]) -> Model:
        supporting_models = [
            model for model in available_models if model.supports_dp(dp)
        ]
        if len(supporting_models) == 0:
            raise ValueError(f"No supporting models found from {available_models}")
        return supporting_models[0]


class CapabilityScoreModel(abc.ABC):
    @abc.abstractmethod
    def score_dp(self, dp: Datapoint) -> float:
        raise NotImplementedError


class PromptedLLMCapabilityScoreModel:
    def __init__(self, model: Model | None = None) -> None:
        if model is None:
            self.model = model

    def score_dp(
        self, dp: Datapoint, examples: list[ScoreDatapoint] | None = None
    ) -> float:
        return (
            self.model.score(
                instruction="Score the task in the datapoint on a scale of 1 (least complex) to 10 (most complex).",
                text=f"----- start task -----\n{dp.model_dump_json()}\n----- end task -----",
                min=1,
                max=10,
                examples=examples,
            )
            / 10.0
        )


def default_request_router() -> RequestRouter:
    return FirstModelRequestRouter()


class RequestRouteDatapoint(BaseModel):
    dp: Datapoint
    capability_score: float
