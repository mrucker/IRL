from abc import ABC, abstractmethod
from typing import Sequence, Tuple
from irl.models import SimModel, State, Action

class Featurizer(ABC):

    @abstractmethod
    def fit_transform(self, X: Sequence[Tuple[State,Action]], y = None) -> Sequence[State]:
        ...

class PostStateFeatures:
    def __init__(self, model: SimModel) -> None:
        self._model = model
    def fit_transform(self, X: Sequence[Tuple[State,Action]], y = None) -> Sequence[State]:
        return list(map(self._model.post_state, *zip(*X)))
