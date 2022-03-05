import gym
import gym.spaces

from typing import overload, Tuple, Optional, Any
from irl.models import GymModel, State, Reward

class GymEnvironment(gym.Env):

    @overload
    def __init__(self, environment: str) -> None: ...

    @overload
    def __init__(self, environment: gym.Env, reward: Reward = None) -> None: ...

    @overload
    def __init__(self, environment: GymModel, reward: Reward) -> None: ...

    def __init__(self, environment, reward: Reward = None):

        if isinstance(environment,str):
            environment = gym.make(environment)

        if isinstance(environment, GymModel):
            self._environment       = environment
            self._actions           = environment.actions(None)
            self._reward            = reward
            self._action_space      = gym.spaces.Discrete(len(self._actions))
            self._observation_space = environment.observation_space

        if isinstance(environment, gym.Env):
            self._environment      = environment
            self._reward           = reward
            self._action_space      = environment.action_space
            self._observation_space = environment.observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

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
            reward      = self._reward([(self._state, self._actions[action])])[0]
            terminal    = self._environment.is_terminal(state)
            self._state = state    
            return state, reward, terminal, {}
