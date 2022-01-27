
import unittest
import io
import math

from itertools import product
from timeit import timeit

import pandas as pd #type: ignore

from combat.representations import Reward, Episode

from combat.domains import phase1
from combat.domains.phase1.data import _encode_categories, _load_lines, load_files

class TestReward(Reward):
    def observe(self, state = None, states = None):
        return 1 if state else [1] * len(states)

class State_Tests(unittest.TestCase):

    def test_set_x_y(self):
        state = phase1.State("a", (0,0,0.5), 2, 4, ((1,2,3),), ((4,5,6),(7,8,9)))

        expected_state = phase1.State("a", (3,4,0.5), 2, 4, ((1,2,3),), ((4,5,6),(7,8,9)))
        actual_state   = state.set_x_y(3,4)

        self.assertEqual(expected_state, actual_state)

    def test_gridify(self):
        state = phase1.State("a", (1,2/3,0.5), 2, 4, ((1,1,3),), ((1/3,3/4,6),(2/3,0,9)))

        expected_state = phase1.State("a", (1,3/4,0.5), 2, 4, ((1,1,3),), ((1/3,3/4,6),(2/3,0,9)))
        actual_state   = state.gridify(5)

        self.assertEqual(expected_state, actual_state)

    def test_round(self):
        state = phase1.State("a", (1,2/3,0.5), 2, 4.222, ((1,1,3),), ((1/3,3/4,6),(2/3,0,9)))

        expected_state = phase1.State("a", (1,0.67,0.5), 2, 4.22, ((1,1,3),), ((.33,.75,6),(.67,0,9)))
        actual_state   = round(state,2)

        self.assertEqual(expected_state, actual_state)

class ContinuousEnvironment_Tests(unittest.TestCase):

    def test_state_actions_transition(self):
        state0 = phase1.State("a", (1,1,0), 0, 0, ((1,1,0),), ((2,2,0),))
        state1 = phase1.State("a", (1,1,0), 1, 4, ((2,2,0),), ((3,3,0),))
        state2 = phase1.State("a", (2,2,0), 2, 8, ((3,3,0),), ((4,4,0),))

        agents = { "a":Episode([state0,state1,state2], [(0,0), (1,1)])}

        expected_state_0 = phase1.State("a", (0,0,0  ), 1, 4, ((2,2,0),), ((3,3,0),))
        expected_state_1 = phase1.State("a", (-1,-1,0), 2, 8, ((3,3,0),), ((4,4,0),))

        dynamics = phase1.ContinuousEnvironment(agents)

        dynamics._state = state0
        state, r, terminal, _ = dynamics.step(0)
        self.assertEqual(state, expected_state_0)
        self.assertEqual(terminal, False)

        state, r, terminal, _ = dynamics.step(0)
        self.assertEqual(state, expected_state_1)
        self.assertEqual(terminal, True)

    def test_rand_init_state(self):

        state0 = phase1.State("a", (1,1,0), 0, 0, ((1,1,0),), ((2,2,0),))
        state1 = phase1.State("a", (1,1,0), 1, 4, ((2,2,0),), ((3,3,0),))
        state2 = phase1.State("a", (3,3,0), 2, 8, ((3,3,0),), ((4,4,0),))

        agents = { "a":Episode([state0,state1,state2], [(0,0), (1,1)])}

        a = phase1.ContinuousEnvironment(agents)

        observed_states = []

        for i in range(100):
            a.reset()
            observed_states.append(a._state)

        for state in observed_states:
            self.assertIn(state, [state0,state1])            

class DiscreteEnvironment_Tests(unittest.TestCase):

    def test_S1(self):

        state0 = phase1.State("a", (0,0,0), 0, 0, ((1,1,0),), ((2,2,0),))
        state1 = phase1.State("a", (1,1,0), 1, 4, ((2,2,0),), ((3,3,0),))
        state2 = phase1.State("a", (1,1,0), 2, 8, ((3,3,0),), ((4,4,0),))

        expert_a = Episode([state0,state1,state2], [(1,1),(0,0)])

        agents = { "a":expert_a, }

        dynamics = phase1.DiscreteEnvironment(agents, 3)

        self.assertEqual(len(dynamics.S), 3*3**2)

        for i,j in product(range(3), range(3)):
            for state in [state0,state1,state2]:
                self.assertIn(state.set_x_y(i/2, j/2), dynamics.S)

    def test_S2(self):

        expert_a = Episode(
            [
             phase1.State("a", (0,0,0), 0, 0, ((1,1,0),), ((2,2,0),)),
             phase1.State("a", (1,1,0), 1, 4, ((2,2,0),), ((3,3,0),)),
             phase1.State("a", (1,1,0), 2, 8, ((3,3,0),), ((4,4,0),))
            ],
            [
             (1,1),
             (0,0),
            ]
        )

        expert_b = Episode(
            [
             phase1.State("b", (1,0,0), 0, 0, ((1,1,0),), ((2,2,0),)),
             phase1.State("b", (1,0,1), 1, 4, ((2,2,0),), ((3,3,0),)),
            ],
            [
             (-1,0),
            ]
        )

        agents = { "a":expert_a, "b":expert_b }

        dynamics = phase1.DiscreteEnvironment(agents, 75)

        self.assertEqual(len(dynamics.S), 5*(75*75))

    def test_A(self):
        state0 = phase1.State("a", (0,0,0), 0, 0, ((1,1,0),), ((2,2,0),))
        state1 = phase1.State("a", (1,1,0), 1, 4, ((2,2,0),), ((3,3,0),))
        state2 = phase1.State("a", (1,1,0), 2, 8, ((3,3,0),), ((4,4,0),))

        expert_a = Episode([state0,state1,state2], [(1,1),(0,0)])

        agents = { "a":expert_a, }

        dynamics = phase1.DiscreteEnvironment(agents, 4)

        self.assertEqual(len(dynamics.A), 9)

        for i,j in product([-1,0,1], [-1,0,1]):
            self.assertIn((i/3, j/3), dynamics._actions)

    def test_transition_tensor(self):

        state0 = phase1.State("a", (0,0,0), 0, 0, ((1,1,0),), ((2,2,0),))
        state1 = phase1.State("a", (1,1,0), 1, 4, ((2,2,0),), ((3,3,0),))

        agents = { "a":Episode([state0,state1], [(1,1)])}

        dynamics = phase1.DiscreteEnvironment(agents, 2)

        sparse_transition = dynamics.transition_tensor
        dense_transition  = dynamics.transition_tensor.to_dense()

        self.assertEqual(36, sparse_transition._nnz())

        self.assertEqual(9, dense_transition.shape[0])
        self.assertEqual(8, dense_transition.shape[1])
        self.assertEqual(8, dense_transition.shape[2])

        self.assertEqual(dense_transition.sum(), 36)

        ##first we check summary values for whole states and actions
        ############################################################

        #all time 1 states have a 100% transition prob for all actions
        for a_i in range(len(dynamics.A)):
            self.assertEqual(dense_transition[a_i,0,:].sum(), 1)
            self.assertEqual(dense_transition[a_i,1,:].sum(), 1)
            self.assertEqual(dense_transition[a_i,2,:].sum(), 1)
            self.assertEqual(dense_transition[a_i,3,:].sum(), 1)

        #all time 2 states have a 0% transition prob for all actions
        self.assertEqual(dense_transition[:,4:8,:].sum(), 0)

        ##then we spot check 3 of the 9 actions
        #######################################

        #all time 1 states transition correctly given action 0 (move down and left)
        self.assertEqual(dense_transition[0,0,4], 1)
        self.assertEqual(dense_transition[0,1,4], 1)
        self.assertEqual(dense_transition[0,2,4], 1)
        self.assertEqual(dense_transition[0,3,4], 1)

        #all time 1 states transition correctly given action 1 (move left)
        self.assertEqual(dense_transition[1,0,4], 1)
        self.assertEqual(dense_transition[1,1,5], 1)
        self.assertEqual(dense_transition[1,2,4], 1)
        self.assertEqual(dense_transition[1,3,5], 1)

        #all time 1 states transition correctly given action 0 (stay in place)
        self.assertEqual(dense_transition[4,0,4], 1)
        self.assertEqual(dense_transition[4,1,5], 1)
        self.assertEqual(dense_transition[4,2,6], 1)
        self.assertEqual(dense_transition[4,3,7], 1)

    def test_transition_tensor2(self):
        
        state0 = phase1.State("a", (0,0,0), 0, 0, ((1,1,0),), ((2,2,0),))
        state1 = phase1.State("a", (1,1,0), 1, 4, ((2,2,0),), ((3,3,0),))

        agents = { "a":Episode([state0,state1], [(1,1)])}

        dynamics = phase1.DiscreteEnvironment(agents, 4)

        sparse_transition = dynamics.transition_tensor

    def test_transition1(self):
        
        state0 = phase1.State("a", (0,0,0), 0, 0, ((1,1,0),), ((2,2,0),))
        state1 = phase1.State("a", (1,1,0), 1, 4, ((2,2,0),), ((3,3,0),))

        agents = { "a":Episode([state0,state1], [(1,1)])}

        dynamics = phase1.DiscreteEnvironment(agents, 2)

        expected_state_1 = phase1.State("a", (0,0,0), 1, 4, ((2,2,0),), ((3,3,0),))
        expected_state_2 = phase1.State("a", (0,1,0), 1, 4, ((2,2,0),), ((3,3,0),))
        expected_state_3 = phase1.State("a", (1,1,0), 1, 4, ((2,2,0),), ((3,3,0),))

        actual_state_1 = dynamics._transition(state0, (-1,-1))
        actual_state_2 = dynamics._transition(state0, (-1, 1))
        actual_state_3 = dynamics._transition(state0, ( 2, 2))

        self.assertEqual(expected_state_1, actual_state_1)
        self.assertEqual(expected_state_2, actual_state_2)
        self.assertEqual(expected_state_3, actual_state_3)

    def test_transition2(self):
        
        state0 = phase1.State("a", (0,0,0), 0, 0, ((1,1,0),), ((2,2,0),))
        state1 = phase1.State("a", (1,1,0), 1, 4, ((2,2,0),), ((3,3,0),))

        agents = { "a":Episode([state0,state1], [(1,1)])}

        dynamics = phase1.DiscreteEnvironment(agents, 4)

        expected_state_1 = phase1.State("a", (0  , 0  , 0), 1, 4, ((2,2,0),), ((3,3,0),))
        expected_state_2 = phase1.State("a", (0  , 2/3, 0), 1, 4, ((2,2,0),), ((3,3,0),))
        expected_state_3 = phase1.State("a", (1  , 1  , 0), 1, 4, ((2,2,0),), ((3,3,0),))
        expected_state_4 = phase1.State("a", (1/3, 1/3, 0), 1, 4, ((2,2,0),), ((3,3,0),))

        actual_state_1 = dynamics._transition(state0, (-1,-1))
        actual_state_2 = dynamics._transition(state0, (-1,0.5))
        actual_state_3 = dynamics._transition(state0, (2,2))
        actual_state_4 = dynamics._transition(state0, (1/3,1/3))

        self.assertEqual(expected_state_1, actual_state_1)
        self.assertEqual(expected_state_2, actual_state_2)
        self.assertEqual(expected_state_3, actual_state_3)
        self.assertEqual(expected_state_4, actual_state_4)

    def test_transition3(self):

        state0 = phase1.State("a", (  0,   0, 0), 0, 0, ((1,1,0),), ((2,2,0),))
        state1 = phase1.State("a", (1/3, 2/3, 0), 1, 4, ((2,2,0),), ((3,3,0),))
        state2 = phase1.State("a", (1/3, 2/3, 0), 2, 8, ((3,3,0),), ((4,4,0),))

        agents = { "a":Episode([state0,state1,state2], [(1/3,2/3),(0,0)])}

        dynamics = phase1.DiscreteEnvironment(agents, 4)

        expected_states = [
            phase1.State("a", (0  , 0  , 0), 2, 8, ((3,3,0),), ((4,4,0),)),
            phase1.State("a", (1/3, 1  , 0), 2, 8, ((3,3,0),), ((4,4,0),)),
            phase1.State("a", (  1, 1/3, 0), 2, 8, ((3,3,0),), ((4,4,0),)),
            phase1.State("a", (2/3,   1, 0), 2, 8, ((3,3,0),), ((4,4,0),))
        ]

        actual_states = [ dynamics._transition(state1, a) for a in [(-2/3,-2/3),(0,2/3),(2/3,-1/3),(1/3,1/3)] ]

        self.assertEqual(expected_states, actual_states)

    def test_performance(self):
        state0   = phase1.State("a", (0,0,0), 0, 0, ((1,1,0),), ((2,2,0),))
        state1   = phase1.State("a", (1,1,0), 1, 4, ((2,2,0),), ((3,3,0),))
        agents   = { "a":Episode([state0,state1], [(1,1)])}
        dynamics = phase1.DiscreteEnvironment(agents, 2)

        
        time = timeit(lambda: dynamics._transition(state0, (-1,-1)), number=10000)

        #I think there is a lot of room for performance improvements. PB=0.02
        print(time)

class Model_Tests(unittest.TestCase):
    def test_from_match_no_missing1_continuous(self):
        rows = [
            {"record_type":"position", "time":0, "name":"p1", "side":"opfor", "x":1, "y":2 , "z":3, "heading":0  , "weapon":"gun", "shooterStance":"STAND", "health":0},
            {"record_type":"position", "time":1, "name":"p1", "side":"opfor", "x":2, "y":2 , "z":3, "heading":90 , "weapon":"gun", "shooterStance":"STAND", "health":0},
            {"record_type":"position", "time":2, "name":"p1", "side":"opfor", "x":2, "y":11, "z":3, "heading":180, "weapon":"gun", "shooterStance":"STAND", "health":0},
        ]

        expected_states  = [
            phase1.State("p1", (  0,0,0), 0, 0, (), ()),
            phase1.State("p1", (1/9,0,0), 1, 1, (), ()),
            phase1.State("p1", (1/9,1,0), 2, 2, (), ())
        ]

        expected_actions = [ 7, 5]

        expected_episode = Episode(expected_states, expected_actions)
        actual_episode   = phase1.Model.from_frame(pd.DataFrame(rows), 1, "p1").expert_episodes[0]

        self.assertEqual(actual_episode.states, expected_episode.states)
        self.assertEqual(actual_episode.actions, expected_episode.actions)

    def test_from_match_no_missing2_continuous(self):
        rows = [
            {"record_type":"position", "time":0, "name":"p1", "side":"opfor", "x":1, "y":2 , "z":3, "heading":0  , "weapon":"gun", "shooterStance":"STAND", "health":0},
            {"record_type":"position", "time":1, "name":"p1", "side":"opfor", "x":2, "y":2 , "z":3, "heading":90 , "weapon":"gun", "shooterStance":"STAND", "health":0},
            {"record_type":"position", "time":2, "name":"p1", "side":"opfor", "x":2, "y":10, "z":3, "heading":180, "weapon":"gun", "shooterStance":"STAND", "health":0},
        ]

        expected_states  = [
            phase1.State("p1", (  0,0,0), 0, 0, (), ()),
            phase1.State("p1", (1/8,0,0), 1, 1, (), ()),
            phase1.State("p1", (1/8,1,0), 2, 2, (), ())
        ]

        expected_actions = [ 7, 5]
        expected_episode = Episode(expected_states, expected_actions)
        actual_episode   = phase1.Model.from_frame(pd.DataFrame(rows), 1, "p1", 9).expert_episodes[0]

        self.assertEqual(actual_episode.states, expected_episode.states)
        self.assertEqual(actual_episode.actions, expected_episode.actions)

    def test_from_match_dies_continuous(self):

        rows = [
            {"record_type":"position", "time":0, "name":"p1", "side":"opfor", "x":1, "y":2 , "z":3, "heading":0  , "weapon":"gun", "shooterStance":"STAND", "health":0},
            {"record_type":"position", "time":1, "name":"p1", "side":"opfor", "x":2, "y":2 , "z":3, "heading":90 , "weapon":"gun", "shooterStance":"STAND", "health":1},
            {"record_type":"position", "time":2, "name":"p1", "side":"opfor", "x":2, "y":10, "z":3, "heading":180, "weapon":"gun", "shooterStance":"STAND", "health":1},
        ]

        expected_states  = [
            phase1.State("p1", (  0,0,0), 0, 0, (), ()),
            phase1.State("p1", (1/8,0,1), 1, 1, (), ())
        ]

        expected_actions = [ 7 ]

        expected_episode = Episode(expected_states, expected_actions)
        actual_episode   = phase1.Model.from_frame(pd.DataFrame(rows), 1, "p1").expert_episodes[0]

        self.assertEqual(actual_episode.states, expected_episode.states)
        self.assertEqual(actual_episode.actions, expected_episode.actions)

    def test_from_match_missing_middle_continuous(self):
        rows = [
            {"record_type":"position", "time":0, "name":"p1", "side":"opfor", "x":1, "y":2 , "z":3, "heading":0  , "weapon":"gun", "shooterStance":"STAND", "health":0},
            {"record_type":"position", "time":8, "name":"p1", "side":"opfor", "x":2, "y":10, "z":3, "heading":180, "weapon":"gun", "shooterStance":"STAND", "health":0},
        ]

        expected_states  = [
            phase1.State("p1", (0   ,   0, 0), 0, 0, (), ()),
            phase1.State("p1", (1/16, 1/2, 0), 1, 4, (), ()),
            phase1.State("p1", ( 1/8,   1, 0), 2, 8, (), ())
        ]

        expected_actions = [ 5, 5 ]

        expected_episode = Episode(expected_states, expected_actions)
        actual_episode   = phase1.Model.from_frame(pd.DataFrame(rows), 1/4, "p1").expert_episodes[0]

        self.assertEqual(actual_episode.states, expected_episode.states)
        self.assertEqual(actual_episode.actions, expected_episode.actions)

    def test_from_match_missing_beginning_continuous(self):
        rows = [
            {"record_type":"position", "time":1, "name":"p1", "side":"opfor", "x":1, "y":2 , "z":3, "heading":0  , "weapon":"gun", "shooterStance":"STAND", "health":0},
            {"record_type":"position", "time":2, "name":"p1", "side":"opfor", "x":2, "y":10, "z":3, "heading":180, "weapon":"gun", "shooterStance":"STAND", "health":0},
        ]

        expected_states  = [
            phase1.State("p1", (  0,   0, 0), 0, 0, (), ()),
            phase1.State("p1", (  0,   0, 0), 1, 1, (), ()),
            phase1.State("p1", (1/8, 8/8, 0), 2, 2, (), ())
        ]

        expected_actions = [ 4, 8 ]
        
        expected_episode = Episode(expected_states, expected_actions)
        actual_episode   = phase1.Model.from_frame(pd.DataFrame(rows), 1, "p1").expert_episodes[0]

        self.assertEqual(actual_episode.states, expected_episode.states)
        self.assertEqual(actual_episode.actions, expected_episode.actions)

    def test_to_artists_continuous(self):

        rows = [
            {"record_type":"position", "time":0, "name":"p1", "side":"opfor", "x":0, "y":0, "z":3, "heading":0  , "weapon":"gun", "shooterStance":"STAND", "health":0},
            {"record_type":"position", "time":1, "name":"p1", "side":"opfor", "x":0, "y":1, "z":3, "heading":90 , "weapon":"gun", "shooterStance":"STAND", "health":0},
            {"record_type":"position", "time":2, "name":"p1", "side":"opfor", "x":1, "y":1, "z":3, "heading":180, "weapon":"gun", "shooterStance":"STAND", "health":0},
        ]

        reward  = TestReward()
        model   = phase1.Model.from_frame(pd.DataFrame(rows), 1, "p1")
        artists = model.to_animated_artists(model.expert_episodes[0], reward)

        #0 other artist + 1 self artist + 1 time artist + 1 reward image
        self.assertEqual(len(artists[0]), 3)
        self.assertEqual(len(artists[1]), 3)

    def test_from_match_no_missing1_discrete(self):
        rows = [
            {"record_type":"position", "time":0, "name":"p1", "side":"opfor", "x":1, "y":2 , "z":3, "heading":0  , "weapon":"gun", "shooterStance":"STAND", "health":0},
            {"record_type":"position", "time":1, "name":"p1", "side":"opfor", "x":2, "y":2 , "z":3, "heading":90 , "weapon":"gun", "shooterStance":"STAND", "health":0},
            {"record_type":"position", "time":2, "name":"p1", "side":"opfor", "x":2, "y":10, "z":3, "heading":180, "weapon":"gun", "shooterStance":"STAND", "health":0},
        ]

        expected_states  = [
            phase1.State("p1", (0,0,0), 0, 0, (), ()),
            phase1.State("p1", (0,0,0), 1, 1, (), ()),
            phase1.State("p1", (0,1,0), 2, 2, (), ())
        ]

        expected_actions = [ 4, 5]

        expected_episode = Episode(expected_states, expected_actions)
        actual_episode   = phase1.Model.from_frame(pd.DataFrame(rows), 1, "p1", 2).expert_episodes[0]

        self.assertEqual(actual_episode.states, expected_episode.states)
        self.assertEqual(actual_episode.actions, expected_episode.actions)

    def test_from_match_no_missing2_discrete(self):
        rows = [
            {"record_type":"position", "time":0, "name":"p1", "side":"opfor", "x":1, "y":2 , "z":3, "heading":0  , "weapon":"gun", "shooterStance":"STAND", "health":0},
            {"record_type":"position", "time":1, "name":"p1", "side":"opfor", "x":2, "y":2 , "z":3, "heading":90 , "weapon":"gun", "shooterStance":"STAND", "health":0},
            {"record_type":"position", "time":2, "name":"p1", "side":"opfor", "x":2, "y":10, "z":3, "heading":180, "weapon":"gun", "shooterStance":"STAND", "health":0},
        ]

        expected_states  = [
            phase1.State("p1", (  0,0,0), 0, 0, (), ()),
            phase1.State("p1", (1/8,0,0), 1, 1, (), ()),
            phase1.State("p1", (1/8,1,0), 2, 2, (), ())
        ]

        expected_actions = [ 7, 5]

        expected_episode = Episode(expected_states, expected_actions)
        actual_episode   = phase1.Model.from_frame(pd.DataFrame(rows), 1, "p1", 9).expert_episodes[0]

        self.assertEqual(actual_episode.states, expected_episode.states)
        self.assertEqual(actual_episode.actions, expected_episode.actions)

    def test_from_match_dies_discrete(self):

        rows = [
            {"record_type":"position", "time":0, "name":"p1", "side":"opfor", "x":1, "y":2 , "z":3, "heading":0  , "weapon":"gun", "shooterStance":"STAND", "health":0},
            {"record_type":"position", "time":1, "name":"p1", "side":"opfor", "x":2, "y":2 , "z":3, "heading":90 , "weapon":"gun", "shooterStance":"STAND", "health":1},
            {"record_type":"position", "time":2, "name":"p1", "side":"opfor", "x":2, "y":10, "z":3, "heading":180, "weapon":"gun", "shooterStance":"STAND", "health":1},
        ]

        expected_states  = [
            phase1.State("p1", (0,0,0), 0, 0, (), ()),
            phase1.State("p1", (0,0,1), 1, 1, (), ()),
        ]

        expected_actions = [ 4 ]

        expected_episode = Episode(expected_states, expected_actions)
        actual_episode   = phase1.Model.from_frame(pd.DataFrame(rows), 1, "p1", 2).expert_episodes[0]

        self.assertEqual(actual_episode.states, expected_episode.states)
        self.assertEqual(actual_episode.actions, expected_episode.actions)

    def test_from_match_missing_middle_discrete(self):
        rows = [
            {"record_type":"position", "time":0, "name":"p1", "side":"opfor", "x":1, "y":2 , "z":3, "heading":0  , "weapon":"gun", "shooterStance":"STAND", "health":0},
            {"record_type":"position", "time":8, "name":"p1", "side":"opfor", "x":2, "y":10, "z":3, "heading":180, "weapon":"gun", "shooterStance":"STAND", "health":0},
        ]

        expected_states  = [
            phase1.State("p1", (0,0,0), 0, 0, (), ()),
            phase1.State("p1", (0,1,0), 1, 4, (), ()),
            phase1.State("p1", (0,1,0), 2, 8, (), ())
        ]

        expected_actions = [ 5, 4]

        expected_episode = Episode(expected_states, expected_actions)
        actual_episode   = phase1.Model.from_frame(pd.DataFrame(rows), 1/4, "p1", 2).expert_episodes[0]

        self.assertEqual(actual_episode.states, expected_episode.states)
        self.assertEqual(actual_episode.actions, expected_episode.actions)

    def test_from_match_missing_beginning_discrete(self):
        rows = [
            {"record_type":"position", "time":1, "name":"p1", "side":"opfor", "x":1, "y":2 , "z":3, "heading":0  , "weapon":"gun", "shooterStance":"STAND", "health":0},
            {"record_type":"position", "time":2, "name":"p1", "side":"opfor", "x":2, "y":10, "z":3, "heading":180, "weapon":"gun", "shooterStance":"STAND", "health":0},
        ]

        expected_states  = [
            phase1.State("p1", (0,0,0), 0, 0, (), ()),
            phase1.State("p1", (0,0,0), 1, 1, (), ()),
            phase1.State("p1", (0,1,0), 2, 2, (), ())
        ]

        expected_actions = [ 4, 5 ]

        expected_episode = Episode(expected_states, expected_actions)
        actual_episode   = phase1.Model.from_frame(pd.DataFrame(rows), 1, "p1", 2).expert_episodes[0]

        self.assertEqual(actual_episode.states, expected_episode.states)
        self.assertEqual(actual_episode.actions, expected_episode.actions)

    def test_to_artists_discrete(self):

        rows = [
            {"record_type":"position", "time":0, "name":"p1", "side":"opfor", "x":1, "y":2 , "z":3, "heading":0  , "weapon":"gun", "shooterStance":"STAND", "health":0},
            {"record_type":"position", "time":1, "name":"p1", "side":"opfor", "x":2, "y":2 , "z":3, "heading":90 , "weapon":"gun", "shooterStance":"STAND", "health":0},
            {"record_type":"position", "time":2, "name":"p1", "side":"opfor", "x":2, "y":10, "z":3, "heading":180, "weapon":"gun", "shooterStance":"STAND", "health":0},
        ]

        model = phase1.Model.from_frame(pd.DataFrame(rows), 1, "p1", 2)
        reward = TestReward()

        actual_artists = model.to_animated_artists(model.expert_episodes[0], reward)

        #0 other artist + 1 self artist + 1 time artist + 1 reward image
        self.assertEqual(len(actual_artists[0]), 3)
        self.assertEqual(len(actual_artists[1]), 3)

class Data_Tests(unittest.TestCase):

    def test_encode_categories(self):

        df = pd.DataFrame({'record_type': ['a','b','a','b','c']})

        self.assertNotEqual(df['record_type'].dtype.name, "category")

        df = _encode_categories([df])[0]

        self.assertEqual(df['record_type'].dtype.name, "category")

    def test_load_lines_group_1(self):

        json_text = """{
            {"people":[{"time":1, "name":"p1", "side":"opfor", "position":[1.1,2.1,3.1], "heading":270.1, "weapon":"gun", "shooterStance":"STAND", "health":1}, {"time":1, "name":"p2", "side":"opfor", "position":[2.2,3.2,4.2], "heading":1.1, "weapon":"gun", "shooterStance":"UNDEFINED", "health":1}]},
            {"unmanned":[{"time":1.5, "name":"u1", "type":"car", "side":"opfor", "position":[5.5,6.6,7.7], "heading":270, "weapon":"big gun"}, {"time":1.5, "name":"u2", "type":"tank", "side":"opfor", "position":[1,2,3], "heading":180, "weapon":""}]}
        }"""

        iterator = _load_lines(io.StringIO(json_text).readlines(), n_group=1) 

        actual_1sts = next(iterator)
        actual_2nds = next(iterator)

        expected_firsts = [
            {'record_type':'people', 'time':1, 'name':'p1', 'side':'opfor', 'x':1.1, 'y':2.1, 'z':3.1, 'heading':270.1, 'weapon':'gun', 'shooterStance':'STAND', 'health':1},
            {'record_type':'people', 'time':1, 'name':'p2', 'side':'opfor', 'x':2.2, 'y':3.2, 'z':4.2, 'heading':1.1, 'weapon':'gun', 'shooterStance':'UNDEFINED', 'health':1}
        ]

        self.assertEqual(actual_1sts.shape[0], len(expected_firsts))
        self.assertEqual(actual_1sts.shape[1], len(expected_firsts[0]))

        for index, key in product(actual_1sts.index, actual_1sts.columns):
            self.assertEqual(actual_1sts.loc[index, key], expected_firsts[index][key])

        expected_seconds = [
            {'record_type':'unmanned', 'time':1.5, 'name':'u1', 'side':'opfor', 'x':5.5, 'y':6.6, 'z':7.7, 'heading':270, 'weapon':'big gun', 'type':'car'},
            {'record_type':'unmanned', 'time':1.5, 'name':'u2', 'side':'opfor', 'x':1.0, 'y':2.0, 'z':3.0, 'heading':180, 'weapon':'', 'type': 'tank'}
        ]

        self.assertEqual(actual_2nds.shape[0], len(expected_seconds))
        self.assertEqual(actual_2nds.shape[1], len(expected_seconds[0]))

        for index, key in product(actual_2nds.index, actual_2nds.columns):
            self.assertEqual(actual_2nds.loc[index, key], expected_seconds[index][key])

    def test_load_lines_group_2(self):

        json_text = """{
            {"people":[{"time":1, "name":"p1", "side":"opfor", "position":[1.1,2.1,3.1], "heading":270.1, "weapon":"gun", "shooterStance":"STAND", "health":1}, {"time":1, "name":"p2", "side":"opfor", "position":[2.2,3.2,4.2], "heading":1.1, "weapon":"gun", "shooterStance":"UNDEFINED", "health":1}]},
            {"unmanned":[{"time":1.5, "name":"u1", "type":"car", "side":"opfor", "position":[5.5,6.6,7.7], "heading":270, "weapon":"big gun"}, {"time":1.5, "name":"u2", "type":"tank", "side":"opfor", "position":[1,2,3], "heading":180, "weapon":""}]}
        }"""

        nan           = float('nan')
        iterator      = _load_lines(io.StringIO(json_text).readlines(), n_group=2) 
        actual_firsts = next(iterator)

        expected_firsts = [
            {'record_type':'people', 'time':1, 'name':'p1', 'side':'opfor', 'x':1.1, 'y':2.1, 'z':3.1, 'heading':270.1, 'weapon':'gun', 'shooterStance':'STAND', 'health':1, 'type':nan},
            {'record_type':'people', 'time':1, 'name':'p2', 'side':'opfor', 'x':2.2, 'y':3.2, 'z':4.2, 'heading':1.1, 'weapon':'gun', 'shooterStance':'UNDEFINED', 'health':1, 'type':nan},
            {'record_type':'unmanned', 'time':1.5, 'name':'u1', 'side':'opfor', 'x':5.5, 'y':6.6, 'z':7.7, 'heading':270, 'weapon':'big gun', 'shooterStance':nan, 'health':nan, 'type':'car'},
            {'record_type':'unmanned', 'time':1.5, 'name':'u2', 'side':'opfor', 'x':1.0, 'y':2.0, 'z':3.0, 'heading':180, 'weapon':'', 'shooterStance':nan, 'health':nan, 'type': 'tank'}
        ]

        self.assertEqual(actual_firsts.shape[0], len(expected_firsts))
        self.assertEqual(actual_firsts.shape[1], len(expected_firsts[0]))

        for index, key in product(actual_firsts.index, actual_firsts.columns):
            actual_value   = actual_firsts.loc[index, key]
            expected_value = expected_firsts[index][key]

            actual_isnan   = isinstance(actual_value, float) and math.isnan(actual_value)
            expected_isnan = isinstance(expected_value, float) and math.isnan(expected_value)

            if  (not actual_isnan) or (not expected_isnan):
                self.assertEqual(actual_value, expected_value)

    def test_load_lines_group_3(self):

        json_text = """{
            {"people":[{"time":1, "name":"p1", "side":"opfor", "position":[1.1,2.1,3.1], "heading":270.1, "weapon":"gun", "shooterStance":"STAND", "health":1}, {"time":1, "name":"p2", "side":"opfor", "position":[2.2,3.2,4.2], "heading":1.1, "weapon":"gun", "shooterStance":"UNDEFINED", "health":1}]},
            {"unmanned":[{"time":1.5, "name":"u1", "type":"car", "side":"opfor", "position":[5.5,6.6,7.7], "heading":270, "weapon":"big gun"}, {"time":1.5, "name":"u2", "type":"tank", "side":"opfor", "position":[1,2,3], "heading":180, "weapon":""}]},
            {"people":[{"time":1, "name":"p1", "side":"opfor", "position":[1.1,2.1,3.1], "heading":270.1, "weapon":"gun", "shooterStance":"STAND", "health":1}, {"time":1, "name":"p2", "side":"opfor", "position":[2.2,3.2,4.2], "heading":1.1, "weapon":"gun", "shooterStance":"UNDEFINED", "health":1}]},
            {"unmanned":[{"time":1.5, "name":"u1", "type":"car", "side":"opfor", "position":[5.5,6.6,7.7], "heading":270, "weapon":"big gun"}, {"time":1.5, "name":"u2", "type":"tank", "side":"opfor", "position":[1,2,3], "heading":180, "weapon":""}]}
        }"""

        nan            = float('nan')
        iterator       = _load_lines(io.StringIO(json_text).readlines(), n_group=3)
        actual_firsts  = next(iterator)
        actual_seconds = next(iterator)

        expected_firsts = [
            {'record_type':'people', 'time':1, 'name':'p1', 'side':'opfor', 'x':1.1, 'y':2.1, 'z':3.1, 'heading':270.1, 'weapon':'gun', 'shooterStance':'STAND', 'health':1, 'type':nan},
            {'record_type':'people', 'time':1, 'name':'p2', 'side':'opfor', 'x':2.2, 'y':3.2, 'z':4.2, 'heading':1.1, 'weapon':'gun', 'shooterStance':'UNDEFINED', 'health':1, 'type':nan},
            {'record_type':'people', 'time':1, 'name':'p1', 'side':'opfor', 'x':1.1, 'y':2.1, 'z':3.1, 'heading':270.1, 'weapon':'gun', 'shooterStance':'STAND', 'health':1, 'type':nan},
            {'record_type':'people', 'time':1, 'name':'p2', 'side':'opfor', 'x':2.2, 'y':3.2, 'z':4.2, 'heading':1.1, 'weapon':'gun', 'shooterStance':'UNDEFINED', 'health':1, 'type':nan},
            {'record_type':'unmanned', 'time':1.5, 'name':'u1', 'side':'opfor', 'x':5.5, 'y':6.6, 'z':7.7, 'heading':270, 'weapon':'big gun', 'shooterStance':nan, 'health':nan, 'type':'car'},
            {'record_type':'unmanned', 'time':1.5, 'name':'u2', 'side':'opfor', 'x':1.0, 'y':2.0, 'z':3.0, 'heading':180, 'weapon':'', 'shooterStance':nan, 'health':nan, 'type': 'tank'}
        ]

        self.assertEqual(actual_firsts.shape[0], len(expected_firsts))
        self.assertEqual(actual_firsts.shape[1], len(expected_firsts[0]))

        for index, key in product(actual_firsts.index, actual_firsts.columns):
            actual_value   = actual_firsts.loc[index, key]
            expected_value = expected_firsts[index][key]

            actual_isnan   = isinstance(actual_value, float) and math.isnan(actual_value)
            expected_isnan = isinstance(expected_value, float) and math.isnan(expected_value)

            if  (not actual_isnan) or (not expected_isnan):
                self.assertEqual(actual_value, expected_value)

        expected_seconds = [
            {'record_type':'unmanned', 'time':1.5, 'name':'u1', 'side':'opfor', 'x':5.5, 'y':6.6, 'z':7.7, 'heading':270, 'weapon':'big gun', 'type':'car'},
            {'record_type':'unmanned', 'time':1.5, 'name':'u2', 'side':'opfor', 'x':1.0, 'y':2.0, 'z':3.0, 'heading':180, 'weapon':'', 'type': 'tank'}
        ]

        self.assertEqual(actual_seconds.shape[0], len(expected_seconds))
        self.assertEqual(actual_seconds.shape[1], len(expected_seconds[0]))

        for index, key in product(actual_seconds.index, actual_seconds.columns):
            actual_value   = actual_seconds.loc[index, key]
            expected_value = expected_seconds[index][key]

            actual_isnan   = isinstance(actual_value, float) and math.isnan(actual_value)
            expected_isnan = isinstance(expected_value, float) and math.isnan(expected_value)

            if  (not actual_isnan) or (not expected_isnan):
                self.assertEqual(actual_value, expected_value)

    def test_load_phase1_match(self):
        f = load_files("phase1", "ambush", "close_assault1")