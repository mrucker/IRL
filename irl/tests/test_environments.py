
import unittest

from typing import Sequence, cast

import torch

from combat.representations import Policy
from combat.environments import MassEnvironment

class TestEnvironment(MassEnvironment):
    def __init__(self, states: torch.Tensor, actions: torch.Tensor, mass: torch.Tensor, init: torch.Tensor = None):
        super().__init__()

        self._states  = states
        self._actions = actions
        self._mass    = mass
        self._init    = init

    def seed(self, seed):
        pass

    @property
    def S(self) -> Sequence[torch.Tensor]:
        return cast(Sequence[torch.Tensor], self._states)

    @property
    def A(self) -> Sequence[torch.Tensor]:
        return cast(Sequence[torch.Tensor], self._actions)

    def transition_mass(self, actions: Sequence[torch.Tensor], state_0s: Sequence[torch.Tensor], state_1s: Sequence[torch.Tensor]) -> torch.Tensor:
        return self._mass.index_select(0, torch.hstack(list(actions))).index_select(1, torch.hstack(list(state_0s))).index_select(2, torch.hstack(list(state_1s)))

    def reset(self):
        
        if self._init is None:
            return super().reset()
        else:
            self._state = self._init
            return self._state

class TestPolicy(Policy):

    def __init__(self, action):
        self._action = action

    def act(self, observation):
        return self._action


class Episode_Tests(unittest.TestCase):
    
    def test_from_policy1(self):
        states  = torch.tensor([[0],[1],[2]])
        actions = torch.tensor([[0],[1]])
        mass    = torch.tensor([[[0,.5,.5],[.5,0,.5],[.5,.5,0]],[[1,0,0],[0,1,0],[0,0,1]]])
        
        dynamics = TestEnvironment(states, actions, mass, states[0])
        policy   = TestPolicy(actions[1, :])

        actual           = dynamics.make_episode(policy,5)
        expected_states  = list(torch.tensor([[0],[0],[0],[0],[0]]))
        expected_actions = list(torch.tensor([[1],[1],[1],[1]]))

        self.assertEqual(actual.states , expected_states)
        self.assertEqual(actual.actions, expected_actions)

    def test_from_policy2(self):
        states  = torch.tensor([[0],[1],[2]])
        actions = torch.tensor([[0],[1]])
        mass    = torch.tensor([[[0,.5,.5],[.5,0,.5],[0,0,0]],[[1,0,0],[0,1,0],[0,0,0]]])

        dynamics = TestEnvironment(states, actions, mass, states[2])
        policy   = TestPolicy(actions[1, :])

        with self.assertRaises(Exception) as ex:
            actual = dynamics.make_episode(policy,5)

        self.assertEqual('A transition was requested for a terminal state', str(ex.exception))
    
    def test_from_policy3(self):
        states  = torch.tensor([[0],[1],[2],])
        actions = torch.tensor([[0],[1]])
        mass    = torch.tensor([[[0,.5,.5],[.5,0,.5],[0,0,0]],[[0,1,0],[0,0,1],[0,0,0]]])
        
        dynamics = TestEnvironment(states, actions, mass, states[0])
        policy   = TestPolicy(actions[1, :])

        actual           = dynamics.make_episode(policy,5)
        expected_states  = list(torch.tensor([[0],[1],[2]]))
        expected_actions = list(torch.tensor([[1],[1]]))

        self.assertEqual(actual.states , expected_states)
        self.assertEqual(actual.actions, expected_actions)

class MassDyamics_Tests(unittest.TestCase):

    def test_is_terminal(self):
        states  = torch.tensor([[0],[1],[2]])
        actions = torch.tensor([[0],[1]])
        mass    = torch.tensor([[[0,1,0],[1,0,0],[0,0,0]],[[0,0,1],[0,0,1],[0,0,0]]])
        
        dynamics = TestEnvironment(states, actions, mass, states[0])

        dynamics.reset()

        state,_,is_terminal,_ = dynamics.step(actions[0])

        self.assertEqual(False, is_terminal)

        state,_,is_terminal,_ = dynamics.step(actions[0])

        self.assertEqual(False, is_terminal)

        state,_,is_terminal,_ = dynamics.step(actions[1])

        self.assertEqual(True, is_terminal)
    
    def test_step(self):
        states  = torch.tensor([[0],[1],[2]])
        actions = torch.tensor([[0],[1]])
        mass    = torch.tensor([[[0,1,0],[1,0,0],[0,0,0]],[[0,0,1],[0,0,1],[0,0,0]]])

        dynamics = TestEnvironment(states, actions, mass, states[0])

        state = dynamics.reset()
        self.assertEqual(torch.tensor([0]), state)

        state,_,is_terminal,_ = dynamics.step(actions[0])
        self.assertEqual(torch.tensor([1]), state)

        state,_,is_terminal,_ = dynamics.step(actions[0])
        self.assertEqual(torch.tensor([0]), state)

        state,_,is_terminal,_ = dynamics.step(actions[1])
        self.assertEqual(torch.tensor([2]), state)

        state = dynamics.reset()
        self.assertEqual(torch.tensor([0]), state)

        state,_,is_terminal,_ = dynamics.step(actions[0])
        self.assertEqual(torch.tensor([1]), state)

        state,_,is_terminal,_ = dynamics.step(actions[1])
        self.assertEqual(torch.tensor([2]), state)