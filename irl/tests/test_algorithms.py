import unittest

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier

from irl.kernels import GaussianKernel
from irl.models import MassModel, Reward, Episode, GymModel
from irl.algorithms import KernelProjection, CascadedSupervised, MaxCausalEnt
from irl.algorithms import ValueIteration, DirectEstimateIteration, StableBaseline

from gym.spaces import Discrete

class TestReward1(Reward):
    def __call__(self, state_actions):
        return [ 0 if s == 0 else -1 for s,a in state_actions]

class TestReward2(Reward):
    def __call__(self, state_actions):
        return [ s/2 for s,a in state_actions]

class SimpleDynamics(MassModel):
    def __init__(self, mass, init = None):        
        self._init = init if init else [1/len(mass[0])]*len(mass[0])
        self._mass = mass

    @property
    def initial_mass(self):
        return self._init

    @property
    def transition_mass(self):
        return self._mass

class ValueIterations_Tests(unittest.TestCase):

    def test_steps_Q(self):
        mass = [[[0,.5,.5],[.5,0,.5],[.5,.5,0]],[[1,0,0],[0,1,0],[0,0,1]]]
        Q    = ValueIteration(1,0,1,"Q").learn_policy(SimpleDynamics(mass),TestReward2())

        self.assertEqual([0.0,0.0], Q(0,[0,1]))
        self.assertEqual([0.5,0.5], Q(1,[0,1]))
        self.assertEqual([1.0,1.0], Q(2,[0,1]))

        Q = ValueIteration(1,0,2,"Q").learn_policy(SimpleDynamics(mass),TestReward2())

        self.assertEqual([0.75,0.00], Q(0,[0,1]))
        self.assertEqual([1.00,1.00], Q(1,[0,1]))
        self.assertEqual([1.25,2.00], Q(2,[0,1]))

    def test_epsilon_policy(self):
        mass   = [[[0,.5,.5],[.5,0,.5],[.5,.5,0]],[[1,0,0],[0,1,0],[0,0,1]]]
        policy = ValueIteration(0.9,0.001).learn_policy(SimpleDynamics(mass),TestReward2())

        self.assertEqual(0, policy(0, [0,1]))
        self.assertEqual(0, policy(1, [0,1]))
        self.assertEqual(1, policy(2, [0,1]))

class DirectEstimateIteration_Tests(unittest.TestCase):

    def test_simple_model1(self):
        mass    = [[[1,0,0],[1,0,0],[0,1,0]],[[0,1,0],[0,0,1],[1,0,0]]]
        policy   = DirectEstimateIteration(10, 10, 4, 2, SVR()).learn_policy(SimpleDynamics(mass),TestReward1())

        self.assertEqual(0, policy(0, [0,1]))
        self.assertEqual(0, policy(1, [0,1]))
        self.assertEqual(1, policy(2, [0,1]))

class KernelProjection_Tests(unittest.TestCase):

    def test_simple_model1(self):
        dynamics       = SimpleDynamics([[[0,.5,.5],[.5,0,.5],[.5,.5,0]],[[1,0,0],[0,1,0],[0,0,1]]])
        true_reward    = TestReward2()
        reward_kernel  = GaussianKernel(.001)
        
        policy_learner = ValueIteration(0.9,0.01)
        reward_learner = KernelProjection(policy_learner, reward_kernel, 1, 0, 10)

        optimal_policy  = policy_learner.learn_policy(dynamics,true_reward)
        expert_episodes = [ Episode.generate(dynamics, optimal_policy, 10) for _ in range(30) ]
        learned_reward  = reward_learner.learn_reward(dynamics, expert_episodes)
        learned_policy  = policy_learner.learn_policy(dynamics, learned_reward)

        self.assertEqual(learned_policy(0, [0,1]), optimal_policy(0, [0,1]))
        self.assertEqual(learned_policy(1, [0,1]), optimal_policy(1, [0,1]))
        self.assertEqual(learned_policy(2, [0,1]), optimal_policy(2, [0,1]))

class CascadedSupervised_Tests(unittest.TestCase):

    def test_simple_model1(self):
        true_reward = TestReward1()
        dynamics    = SimpleDynamics([[[1,0,0],[0,0,1],[0,0,1]],[[0,1,0],[1,0,0],[0,1,0]],[[0,0,1],[0,0,1],[1,0,0]]])

        policy_learner = ValueIteration(0.9,0.01)
        reward_learner = CascadedSupervised(.9, RandomForestClassifier(max_depth=1), SVR())

        optimal_policy  = policy_learner.learn_policy(dynamics,true_reward)
        expert_episodes = [ Episode.generate(dynamics, optimal_policy, 3) for _ in range(50) ]
        learned_reward  = reward_learner.learn_reward(dynamics, expert_episodes)
        learned_policy  = policy_learner.learn_policy(dynamics, learned_reward)

        self.assertEqual(learned_policy(0, [0,1]), optimal_policy(0, [0,1]))
        self.assertEqual(learned_policy(1, [0,1]), optimal_policy(1, [0,1]))
        self.assertEqual(learned_policy(2, [0,1]), optimal_policy(2, [0,1]))

class MaxCausalEntApprximation_Tests(unittest.TestCase):

    def test_simple_model1(self):
        dynamics    = SimpleDynamics([[[0,.5,.5],[.5,0,.5],[.5,.5,0]],[[1,0,0],[0,1,0],[0,0,1]]]) 
        true_reward = TestReward2()
        
        policy_learner = ValueIteration(0.99,0.01)
        reward_learner = MaxCausalEnt()

        optimal_policy  = policy_learner.learn_policy(dynamics,true_reward)
        expert_episodes = [ Episode.generate(dynamics, optimal_policy, 3) for _ in range(50) ]
        learned_reward  = reward_learner.learn_reward(dynamics, expert_episodes)
        learned_policy  = policy_learner.learn_policy(dynamics, learned_reward)

        self.assertEqual(learned_policy(0, [0,1]), optimal_policy(0, [0,1]))
        self.assertEqual(learned_policy(1, [0,1]), optimal_policy(1, [0,1]))
        self.assertEqual(learned_policy(2, [0,1]), optimal_policy(2, [0,1]))

class StableBaselines_Tests(unittest.TestCase):

    def test_simple_model1(self):
        mass    = [[[1,0,0],[1,0,0],[0,1,0]],[[0,1,0],[0,0,1],[1,0,0]]]
        model   = GymModel(SimpleDynamics(mass), Discrete(3))
        policy  = StableBaseline("A2C", 1000, 5).learn_policy(model,TestReward1())

        self.assertEqual(0, policy(0, [0,1]))
        self.assertEqual(0, policy(1, [0,1]))
        self.assertEqual(1, policy(2, [0,1]))
