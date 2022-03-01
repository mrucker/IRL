import random
from abc import abstractmethod, ABC
from typing import Sequence, Any, Tuple

IsTerminal = bool

Action     = Any
State      = Any
Post_State = Any

class Policy(ABC):
    @abstractmethod
    def __call__(self, state: State, actions: Sequence[Action]) -> Action:
        ...

class Reward(ABC):
    @abstractmethod
    def __call__(self, states: Sequence[Tuple[State,Action]]) -> Sequence[float]:
        ...

class SimModel(ABC):
    @abstractmethod 
    def actions(self, state: State) -> Sequence[Action]:
        ...

    @abstractmethod
    def next_state(self, state: State, action: Action) -> State:
        ...

    @abstractmethod
    def post_state(self, state: State, action: Action) -> Post_State:
        ...

    @abstractmethod
    def is_terminal(self, state:State) -> bool:
        ...

    @abstractmethod
    def initial_state(self) -> State:
        ...

class MassModel(SimModel, ABC):

    @property
    @abstractmethod
    def initial_mass(self) -> Sequence[float]:
        ...

    @property
    @abstractmethod
    def transition_mass(self) -> Sequence[Sequence[Sequence[float]]]:
        ...

    def initial_state(self) -> State:
        return random.choice(list(range(len(self.initial_mass))), self.initial_mass)

    def actions(self, state: State) -> Sequence[Action]:
        return list(range(len(self.transition_mass)))

    def next_state(self, state: State, action: Action) -> State:
        return random.choice(list(range(len(self.initial_mass))), self.transition_mass[action][state])

    def post_state(self, state: State , action: Action) -> Post_State:
        return (state,action)

class Episode:

    @staticmethod
    def generate(model: SimModel, policy: Policy, length:int, start:State=None):

        states  = [start or model.initial_state()]
        actions = []

        for _ in range(length-1):
            actions.append(policy(states[-1]))
            states.append(model.next_state(model.post_state(states[-1], actions[-1])))

        return Episode(states,actions) 

    def __init__(self, states: Sequence[State], actions: Sequence[Action]) -> None:

        assert len(states)-1 == len(actions), "We assume that there is a trailing state"

        self.states   = states 
        self.actions  = actions

    @property
    def start(self) -> State:
        return self.states[0]

    def __len__(self):
        return len(self.states)
