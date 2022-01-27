import random

from abc import abstractmethod, ABC
from typing import Sequence, List, Any, Tuple, overload, Optional
from functools import lru_cache as memoized

import gym         #type:ignore
import gym.spaces  #type:ignore
import torch       #type:ignore

from combat.representations import Featurizer, Observation, Action, Policy, Reward, IdentityFeaturizer, Episode

IsTerminal = bool

class SimEnvironment(ABC):

    @abstractmethod
    def reset(self) -> Observation: ...

    @abstractmethod
    def step(self, action:int) -> Tuple[Observation, Optional[float], IsTerminal, Any]: ...

    @abstractmethod
    def actions(self) -> Sequence[Action]: ...

    @abstractmethod
    def seed(self, seed:int) -> None: ...

    def make_episode(self, policy:Policy, length:int = 999) -> Episode:
        
        states :List[Observation] = []
        actions:List[Action     ] = []
        rewards:List[float      ] = []

        states.append(self.reset())

        for t in range(length-1):

            actions.append(policy.act(states[-1]))
            state,r,terminal,_ = self.step(actions[-1])

            rewards.append(r)
            states.append(state)

            if terminal: break

        return Episode(states, actions, rewards)

class MassEnvironment(SimEnvironment):

    @property #type: ignore
    @memoized(maxsize=1)
    def transition_tensor(self) -> torch.Tensor:
        return self.transition_mass(self.A, self.S, self.S)

    @property
    @abstractmethod
    def S(self) -> Sequence[Observation]:
        ...

    @property
    @abstractmethod
    def A(self) -> Sequence[Action]:
        ...

    @abstractmethod
    def transition_mass(self, actions: Sequence[Action], state_0s: Sequence[Observation], state_1s: Sequence[Observation]) -> torch.Tensor:
        ...

    def reset(self):
        self._state = random.choice(self.S)

        if self._is_terminal(self._state):
            raise Exception("An initial state was terminal.")

        return self._state

    def step(self, action) -> Tuple[Observation, Optional[float], IsTerminal, Any]: 

        if self._is_terminal(self._state):
            raise Exception("A transition was requested for a terminal state")

        pmf = self.transition_mass([action], [self._state], self.S).to_dense().squeeze()
        cdf = pmf.cumsum(0)
        idx = (cdf >= torch.rand(())).int().argmax().item()

        self._state = self.S[idx]

        observation = self._state
        is_terminal = self._is_terminal(self._state)
        reward      = None
        info        = {}

        return observation, reward, is_terminal, info 

    def actions(self):
        return self.A

    def _is_terminal(self, state):
        return self.transition_mass(self.A, [state], self.S).sum().item() == 0

class GymEnvironment(SimEnvironment, gym.Env):

    @overload
    def __init__(self, environment: str) -> None: ...

    @overload
    def __init__(self, environment: gym.Env) -> None: ...

    @overload
    def __init__(self, environment: SimEnvironment, observation_space: gym.spaces.Space, observer: Featurizer = IdentityFeaturizer()) -> None: ...

    def __init__(self, environment, observation_space = None, observer = None):

        if isinstance(environment,str):
            environment = gym.make(environment)

        if isinstance(environment, SimEnvironment):
            self._environment      = environment
            self._observer         = observer or IdentityFeaturizer()
            self.action_space      = gym.spaces.Discrete(len(environment.actions()))
            self.observation_space = observation_space

        if isinstance(environment, gym.Env):
            self._environment      = environment
            self._observer         = IdentityFeaturizer()
            self.action_space      = environment.action_space
            self.observation_space = environment.observation_space

    def actions(self) -> Sequence[int]:
        if isinstance(self._environment, gym.Env):
            return list(range(self._environment.action_space.n))
        else:
            return self._environment.actions()

    def reset(self) -> Observation:
        return self._observer.to_features([self._environment.reset()])[0]

    def step(self, action:int) -> Tuple[Observation, Optional[float], IsTerminal, Any]:
        observation, reward, isterminal, info = self._environment.step(action)
        return self._observer.to_features([observation])[0], reward, isterminal, info

    def seed(self, seed:int) -> None:
        self._environment.seed(seed)

class RewardEnvironment(SimEnvironment):

    def __init__(self, environment: SimEnvironment, reward: Reward) -> None:
        self._environment = environment
        self._reward      = reward

    def actions(self) -> Sequence[int]:
        return self._environment.actions()

    def reset(self) -> Observation:
        return self._environment.reset()

    def step(self, action: int) -> Tuple[Observation, float, IsTerminal, Any]:

        observation, reward, isterminal, info = self._environment.step(action)
        
        reward = self._reward.observe(state=observation) if self._reward else reward

        return observation, reward, isterminal, info

    def seed(self, seed:int) -> None:
        self._environment.seed(seed)