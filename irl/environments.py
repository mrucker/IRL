import gym
import gym.spaces

from typing import overload, Sequence, Tuple, Optional, Any
from irl.models import SimModel, State, Reward

class GymEnvironment(gym.Env):

    @overload
    def __init__(self, environment: str) -> None: ...

    @overload
    def __init__(self, environment: gym.Env, reward: Reward = None) -> None: ...

    @overload
    def __init__(self, environment: SimModel, reward: Reward, observation_space: gym.spaces.Space) -> None: ...

    def __init__(self, environment, reward: Reward = None, observation_space: gym.spaces.Space = None):

        if isinstance(environment,str):
            environment = gym.make(environment)

        if isinstance(environment, SimModel):
            self._environment      = environment
            self._actions          = environment.actions()
            self._reward           = reward
            self.action_space      = gym.spaces.Discrete(len(self._actions))
            self.observation_space = observation_space

        if isinstance(environment, gym.Env):
            self._environment      = environment
            self._reward           = reward
            self.action_space      = environment.action_space
            self.observation_space = environment.observation_space

    def actions(self) -> Sequence[int]:
        if isinstance(self._environment, gym.Env):
            return list(range(self._environment.action_space.n))
        else:
            return self._environment.actions()

    def reset(self) -> State:
        if isinstance(self._environment, gym.Env):
            return self.reset()
        else:
            self._state = self._environment.initial_state()
            return self._state

    def step(self, action:int) -> Tuple[State, Optional[float], bool, Any]:
        if isinstance(self._environment, gym.Env):
            state, reward, terminal, info = self._environment.step(action)
            if self._reward is not None: reward = self._reward([ (self._state, action) ])[0]
            return state, reward, terminal, info
        else:
            state       = self._environment.next_state(self._state, self._actions[action])
            reward      = self._reward([(self._state, self._actions[action])])
            terminal    = self._environment.is_terminal(state)
            self._state = state    
            return state, reward, terminal, {}
