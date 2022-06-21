
import unittest
from typing import Sequence

import torch
from irl.models import MassModel, Policy, Episode

class TestModel(MassModel):
    def __init__(self, mass):
        self._mass = mass

    @property
    def initial_mass(self) -> Sequence[float]:
        return [1,0,0]

    @property
    def transition_mass(self) -> Sequence[Sequence[Sequence[float]]]:
        return self._mass

class TestPolicy(Policy):

    def __init__(self, action):
        self._action = action

    def __call__(self, state, actions):
        return self._action

class Episode_Tests(unittest.TestCase):
    
    def test_from_policy1(self):
        mass    = [[[0,.5,.5],[.5,0,.5],[.5,.5,0]],[[1,0,0],[0,1,0],[0,0,1]]]

        dynamics = TestModel(mass)
        policy   = TestPolicy(1)

        actual           = Episode.generate(dynamics,policy,5)
        expected_states  = [0]*5
        expected_actions = [1]*4

        self.assertEqual(actual.states , expected_states)
        self.assertEqual(actual.actions, expected_actions)

    def test_from_policy2(self):
        mass    = [[[0,0,0],[.5,0,.5],[.5,.5,0]],[[0,0,0],[0,1,0],[.5,.5,0]]]

        dynamics = TestModel(mass)
        policy   = TestPolicy(1)

        with self.assertRaises(Exception) as ex:
            actual = Episode.generate(dynamics,policy,5)

        self.assertEqual('A transition was requested for a terminal state', str(ex.exception))

    def test_from_policy3(self):
        mass = [[[0,.5,.5],[.5,0,.5],[0,0,0]],[[0,1,0],[0,0,1],[0,0,0]]]

        dynamics = TestModel(mass)
        policy   = TestPolicy(1)

        actual           = Episode.generate(dynamics, policy,5)
        expected_states  = [0,1,2]
        expected_actions = [1,1]

        self.assertEqual(actual.states , expected_states)
        self.assertEqual(actual.actions, expected_actions)

class MassDyamics_Tests(unittest.TestCase):

    def test_is_terminal(self):
        mass    = torch.tensor([[[0,1,0],[1,0,0],[0,0,0]],[[0,0,1],[0,0,1],[0,0,0]]])
        
        dynamics = TestModel(mass)
        self.assertEqual(False, dynamics.is_terminal(0))
        self.assertEqual(False, dynamics.is_terminal(1))
        self.assertEqual(True , dynamics.is_terminal(2))
    
    def test_step(self):
        mass = [[[0,1,0],[1,0,0],[0,0,0]],[[0,0,1],[0,0,1],[0,0,0]]]

        dynamics = TestModel(mass)

        self.assertEqual(1, dynamics.next_state(0,0))
        self.assertEqual(0, dynamics.next_state(1,0))
        self.assertEqual(2, dynamics.next_state(0,1))
