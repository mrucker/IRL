import unittest

import torch
from sklearn.svm import SVR

from combat.representations import EqualityKernel, Reward
from combat.environments import MassEnvironment
from combat.algorithms import DirectEstimateIteration, ValueIteration, KernelProjection, SklearnRegressor, CascadedSupervised, MaxCausalEnt

class TestReward1(Reward):
    def observe(self, state = None, states = None):
        return -int(state[0]!=0) if state else [-int(s[0]!=0) for s in states]

class TestReward2(Reward):
    def observe(self, state = None, states = None):
        return (state+1).squeeze().item() if state else (states+1).squeeze().tolist()

class TestReward3(Reward):
    def __init__(self, rewards):
        self._rewards = rewards

    def observe(self, state = None, states = None):
        return self._rewards[state].squeeze().item() if state else self._rewards[states].squeeze().tolist()

class TestReward4(Reward):
    def observe(self, state = None, states = None):
        return state+1 if state is not None else [state+1 for state in states]

class SimpleDynamics(MassEnvironment):
    def __init__(self,states, actions, mass, init = None):
        
        super().__init__()
        
        self._init    = init
        self._states  = states
        self._actions = actions
        self._mass    = mass

    def seed(self, seed):
        pass

    @property
    def S(self):
        return self._states

    @property
    def A(self):
        return self._actions

    def transition_mass(self, actions, state_0s, state_1s) -> torch.Tensor:

        mass = torch.zeros(size=(len(actions), len(state_0s), len(state_1s)))

        for ai, action in enumerate(actions):
            for si, state0 in enumerate(state_0s):
                for sii, state1 in enumerate(state_1s):
                    mass[ai,si,sii] = self._mass[action,state0,state1]

        return mass

class DirectEstimateIteration_Tests(unittest.TestCase):

    @unittest.skip("I've been tweaking CascadedSupervised internally making this a somewhat brittle test")
    def test_simple_model1(self):
        states  = torch.tensor([[0],[1],[2]])
        actions = torch.tensor([[0],[1]])
        mass    = torch.tensor([[[1,0,0],[1,0,0],[0,1,0]],[[0,1,0],[0,0,1],[0,0,1]]])
        reward  = TestReward1()
        init    = torch.tensor([[2]])

        learner  = SklearnRegressor(SVR(kernel=EqualityKernel().eval))
        dynamics = SimpleDynamics(states, actions, mass, init)
        policy   = DirectEstimateIteration(4, 10, 8, 3, learner).learn_policy(dynamics,reward)

        self.assertEqual(policy.act(states[0,:]),0)
        self.assertEqual(policy.act(states[1,:]),0)
        self.assertEqual(policy.act(states[2,:]),0)

class ValueIterations_Tests(unittest.TestCase):

    def test_simple_model1(self):
        states  = torch.tensor([[0],[1],[2]])
        actions = torch.tensor([[0],[1]])
        mass    = torch.tensor([[[0,.5,.5],[.5,0,.5],[.5,.5,0]],[[1,0,0],[0,1,0],[0,0,1]]])
        reward  = TestReward2()

        dynamics = SimpleDynamics(states, actions, mass)

        policy = ValueIteration(0.9,0.001).learn_policy(dynamics,reward)

        self.assertEqual(policy.act(states[0,:]),0)
        self.assertEqual(policy.act(states[1,:]),0)
        self.assertEqual(policy.act(states[2,:]),1)

    def test_simple_model2(self):
        states  = torch.tensor([[0],[1],[2]])
        actions = torch.tensor([[0],[1]])
        mass    = torch.tensor([[[0,.5,.5],[.5,0,.5],[.5,.5,0]],[[1,0,0],[0,1,0],[0,0,1]]])
        reward  = TestReward3(torch.tensor([[1],[0],[0]]))

        dynamics = SimpleDynamics(states, actions, mass)
        policy   = ValueIteration(0.9,0.001).learn_policy(dynamics,reward)

        self.assertEqual(policy.act(states[0,:]),1)
        self.assertEqual(policy.act(states[1,:]),0)
        self.assertEqual(policy.act(states[2,:]),0)

    def test_simple_model3(self):

        states  = torch.tensor([[0],[1],[2]])
        actions = torch.tensor([[0],[1]])
        mass    = torch.tensor([[[0,.5,.5],[.5,0,.5],[.5,.5,0]],[[1,0,0],[0,1,0],[0,0,1]]])
        reward  = TestReward3(torch.tensor([[0.68],[-1.73],[1.04]]))

        dynamics = SimpleDynamics(states, actions, mass)

        policy = ValueIteration(0.9,0.001).learn_policy(dynamics,reward)

        self.assertEqual(policy.act(states[0,:]),0)
        self.assertEqual(policy.act(states[1,:]),0)
        self.assertEqual(policy.act(states[2,:]),1)

class KernelProjection_Tests(unittest.TestCase):

    def test_simple_model1(self):
        states      = torch.tensor([[0],[1],[2]])
        actions     = torch.tensor([[0],[1]])
        mass        = torch.tensor([[[0,.5,.5],[.5,0,.5],[.5,.5,0]],[[1,0,0],[0,1,0],[0,0,1]]])
        true_reward = TestReward2()
        kernel      = EqualityKernel()

        dynamics = SimpleDynamics(states, actions, mass)
        learner  = ValueIteration(0.9,0.01)

        optimal_policy = learner.learn_policy(dynamics,true_reward)

        episode_length = 10
        episode_count  = 30
        episodes       = [ dynamics.make_episode(optimal_policy,episode_length) for _ in range(episode_count) ]

        learned_reward = KernelProjection(learner, kernel, 0.9, 0.01, 100, False).learn_reward(dynamics, episodes)[0]

        optimal_policy = learner.learn_policy(dynamics, true_reward)
        learned_policy = learner.learn_policy(dynamics, learned_reward)

        self.assertEqual(learned_policy.act(states[0,:]), optimal_policy.act(states[0,:]))
        self.assertEqual(learned_policy.act(states[1,:]), optimal_policy.act(states[1,:]))
        self.assertEqual(learned_policy.act(states[2,:]), optimal_policy.act(states[2,:]))

class CascadedSupervised_Tests(unittest.TestCase):

    def test_simple_model1(self):
        states      = [0,1,2]
        actions     = [0,1]
        mass        = torch.tensor([[[0,.5,.5],[.5,0,.5],[.5,.5,0]],[[1,0,0],[0,1,0],[0,0,1]]])
        true_reward = TestReward4()

        dynamics = SimpleDynamics(states, actions, mass)
        learner  = ValueIteration(0.99,0.01)

        optimal_policy = learner.learn_policy(dynamics,true_reward)

        episode_length = 10
        episode_count  = 30
        episodes       = [ dynamics.make_episode(optimal_policy,episode_length) for _ in range(episode_count) ]

        learned_reward = CascadedSupervised().learn_reward(dynamics, episodes)

        optimal_policy = learner.learn_policy(dynamics, true_reward)
        learned_policy = learner.learn_policy(dynamics, learned_reward)

        self.assertEqual(learned_policy.act(states[0]), optimal_policy.act(states[0]))
        self.assertEqual(learned_policy.act(states[1]), optimal_policy.act(states[1]))
        self.assertEqual(learned_policy.act(states[2]), optimal_policy.act(states[2]))

class MaxCausalEntApprximation_Tests(unittest.TestCase):

    def test_simple_model1(self):
        states      = [0,1,2]
        actions     = [0,1]
        mass        = torch.tensor([[[0,.5,.5],[.5,0,.5],[.5,.5,0]],[[1,0,0],[0,1,0],[0,0,1]]])
        true_reward = TestReward4()

        dynamics = SimpleDynamics(states, actions, mass)
        learner  = ValueIteration(0.99,0.01)

        optimal_policy = learner.learn_policy(dynamics,true_reward)

        episode_length = 10
        episode_count  = 30
        episodes       = [ dynamics.make_episode(optimal_policy,episode_length) for _ in range(episode_count) ]

        learned_reward = MaxCausalEnt().learn_reward(dynamics, episodes)

        optimal_policy = learner.learn_policy(dynamics, true_reward)
        learned_policy = learner.learn_policy(dynamics, learned_reward)

        self.assertEqual(learned_policy.act(states[0]), optimal_policy.act(states[0]))
        self.assertEqual(learned_policy.act(states[1]), optimal_policy.act(states[1]))
        self.assertEqual(learned_policy.act(states[2]), optimal_policy.act(states[2]))
