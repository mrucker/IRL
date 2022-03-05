import random
from abc import abstractmethod, ABC
from typing import Sequence, Any, Tuple
from gym.spaces import Space

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
    def __call__(self, states_actions: Sequence[Tuple[State,Action]]) -> Sequence[float]:
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

    def states(self) -> Sequence[State]:
        return list(range(len(self.transition_mass[0])))

    def actions(self, _: State=None) -> Sequence[Action]:
        return list(range(len(self.transition_mass)))

    @property
    def initial_mass(self) -> Sequence[float]:
        return [1/len(self.states())]*len(self.states())

    @property
    @abstractmethod
    def transition_mass(self) -> Sequence[Sequence[Sequence[float]]]:
        ...

    def initial_state(self) -> State:
        return random.choices(self.states(),self.initial_mass, k=1)[0]

    def next_state(self, state: State, action: Action) -> State:
        if self.is_terminal(state): raise Exception("A transition was requested for a terminal state")
        return random.choices(self.states(), self.transition_mass[action][state])[0]

    def post_state(self, state: State , action: Action) -> Post_State:
        return (state,action)

    def is_terminal(self, state) -> bool:
        state_index = self.states().index(state)

        return 0 == sum([ sum(t[state_index]) for t in self.transition_mass ])

class GymModel(SimModel):
    
    def __init__(self, model: SimModel, observation_space: Space) -> None:
        self._model = model
        self._observation_space = observation_space

    @property
    def observation_space(self) -> Space:
        return self._observation_space

    def actions(self, state: State) -> Sequence[Action]:
        return self._model.actions(state)

    def next_state(self, state: State, action: Action) -> State:
        return self._model.next_state(state, action)

    def post_state(self, state: State, action: Action) -> Post_State:
        return self._model.post_state(state, action)

    def is_terminal(self, state: State) -> bool:
        return self._model.is_terminal(state)

    def initial_state(self) -> State:
        return self._model.initial_state()

class Episode:

    @staticmethod
    def generate(model: SimModel, policy: Policy, length:int, start:State=None):

        states  = [start or model.initial_state()]
        actions = []

        for _ in range(length-1):
            actions.append(policy(states[-1], model.actions(states[-1])))
            states.append(model.next_state(states[-1], actions[-1]))
            if model.is_terminal(states[-1]): break

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
