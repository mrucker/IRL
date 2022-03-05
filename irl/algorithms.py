import collections
import random
import time
import math

from abc import ABC, abstractmethod
from itertools import chain, repeat
from typing import Sequence, List, Tuple, Optional, Literal, Union, Type

import numpy as np
import torch

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env

import sklearn.base
import sklearn.utils
import sklearn.exceptions
import sklearn.model_selection
import sklearn.ensemble
import sklearn.linear_model
import sklearn.utils.extmath
import sklearn.neural_network
import sklearn.neural_network._base
import sklearn.neural_network._stochastic_optimizers
import sklearn.kernel_approximation

from irl.models import State, Action, Policy, Reward, MassModel, SimModel, GymModel, Episode
from irl.kernels import Kernel, KernelVector
from irl.environments import GymEnvironment

class PolicyLearner(ABC):
    
    @abstractmethod
    def learn_policy(self, model, reward: Reward) -> Policy:
        ...

class RewardLearner(ABC):
    @abstractmethod
    def learn_reward(self, model, episodes: Sequence[Episode]) -> Reward:
        ...

class Regressor(ABC):

    @abstractmethod
    def fit(self, X, y) -> 'Regressor':
        ...

    @abstractmethod
    def predict(self, X) -> Sequence[float]:
        ...

class ValueIteration(PolicyLearner):

    #Classic, but requires a very specific model

    def __init__(self, 
        gamma: float=0.9,
        epsilon: float=0.1, 
        steps: int = None,
        mode: Literal["policy","Q"] = "policy") -> None:

        assert epsilon < 1, "if epsilon is >= 1 no learning will occur"

        self._gamma   = gamma
        self._epsilon = epsilon
        self._steps   = steps or float('inf')
        self._mode    = mode

    def learn_policy(self, model: MassModel, reward: Reward) -> Policy:
        S = model.states()
        A = model.actions()
        R = torch.full((len(A),len(S),1), 0).float()
        Q = torch.full((len(A),len(S),1), 0).float()
        T = torch.tensor(model.transition_mass).float()

        for a_i in range(len(A)):
            R[a_i,:] = torch.tensor(reward(list(zip(S, repeat(a_i))))).unsqueeze(1)

        assert R.shape[0] == len(A)
        assert R.shape[1] == len(S)

        #scale all rewards to length of 1 so that epsilon is meaningful regardless
        #of the given rewards. This scaling has no effect on the optimal policy.

        min = torch.min(R)
        max = torch.max(R)
        R   = (R-min) / (max-min)

        step = 0
        old_V, new_V = -torch.max(R, 0, keepdim=False)[0], torch.zeros((len(S),1))

        while (new_V-old_V).abs().max() > self._epsilon and step < self._steps:
            step += 1

            Q[:] = R[:] + torch.bmm(T, (self._gamma * new_V).unsqueeze(0).repeat(len(A),1,1))

            old_V, new_V = new_V, torch.max(Q, 0, keepdim=False)[0]

        if self._mode == "policy":
            return lambda state, actions, Q=Q: actions[int(torch.argmax(Q[actions,state,:]))]
        else:
            return lambda state, actions, Q=Q: [ Q[a,state,0] for a in actions ]

class DirectEstimateIteration(PolicyLearner):

    #Very similar to MCTS

    class NullQ(Regressor):

        def fit(self, X, y) -> 'Regressor':
            return self 

        def predict(self, X) -> torch.Tensor:
            return torch.tensor([0]* len(X)).float()

    class GreedyPolicy(Policy):
        def __init__(self, Q: Regressor) -> None:
            self._Q = Q

        def __call__(self, state: State, actions: Sequence[Action]) -> Action:
            q_values = torch.tensor(self._Q.predict(list(zip(repeat(state), actions))))
            argmaxes = (q_values == q_values.max()).int().nonzero(as_tuple=True)[0].tolist()
            return actions[random.choice(argmaxes)]

    def __init__(self, 
        n_iters: int,
        n_episodes: int,
        n_steps: int,
        n_target: Optional[int],
        regressor: Regressor,
        bootstrap: float = 0,
        start_policy: Union[Literal['softmax'], Literal['greedy'], Literal['epsilon']] = 'softmax',
        episode_policy: Union[Literal['softmax'], Literal['greedy'], Literal['epsilon']] = 'greedy',
        previous_samples: Optional[int] = None,
        terminal_zero: bool = True):

        self._regressor  = regressor #The regressor to use when learning our q-function
        self._n_iters    = n_iters
        self._n_episodes = n_episodes
        self._n_steps    = n_steps
        self._n_target   = n_target

        self._bootstrap        = bootstrap #what percent of the bootstrap value to use (i.e., bootstrap in [0,1])
        self._start_policy     = start_policy #the policy to use to select the first action in the episodes 
        self._episode_policy   = episode_policy #the policy to use to select all actions after the first
        self._previous_samples = previous_samples if previous_samples is not None else n_iters #how many prev to learn from
        self._terminal_zero    = terminal_zero

    def learn_policy(self, dynamics: SimModel, reward: Reward) -> Policy:

        times = [0.,0.,0.]

        def policy(s0, Q: Regressor, policy_type: Literal['softmax', 'greedy', 'epsilon']):
            
            a0s = dynamics.actions(s0)

            if policy_type == 'epsilon':
                return random.choice(a0s) if random.random() < 0.5 else policy(s0, Q, 'greedy')

            qs = torch.tensor(Q.predict(list(zip(repeat(s0), a0s))))

            if policy_type == 'softmax':
                es = torch.exp(qs-qs.max())
                ps = es/sum(es)
                return random.choices(a0s, ps, k=1)[0]

            if policy_type == 'greedy':
                max_indexes = (qs == qs.max()).int().nonzero(as_tuple=True)[0].tolist()
                return a0s[random.choice(max_indexes)]

            raise Exception(f"Unrecognized policy: {policy_type}")

        def bootstrap(S: Sequence[State], A: Sequence[Action], Q: Regressor, end: int):
            if end >= len(A) or S[end] is None or A[end] is None:
                return None
            else:
                return torch.tensor(Q.predict([(S[end],A[end])]))[0].item()

        Q = DirectEstimateIteration.NullQ()

        iter_episodes: List[Tuple[List[Tuple[State,Action]], List[float], List[int], List[int], Sequence[Tuple[State,Action,str]]]] = []

        for _ in range(self._n_iters):
            SS: List[List[State]]     = []
            AA: List[List[Action]]    = []
            RR: List[Sequence[float]] = []

            start_time = time.time()
            for _ in range(self._n_episodes):

                S: List[State ] = []
                A: List[Action] = []
                R: List[float ] = []

                S.append(dynamics.initial_state())
                A.append(policy(S[-1], Q, self._start_policy))

                for _ in range(1, self._n_steps):

                    s = dynamics.next_state(S[-1], A[-1])
                    r = reward([(S[-1],A[-1])])[0]

                    R.append(r)
                    S.append(s)

                    if dynamics.is_terminal(s) and self._terminal_zero:
                        A.extend([None]*(self._n_steps-len(A)  ))
                        S.extend([None]*(self._n_steps-len(S)  ))
                        R.extend([0]   *(self._n_steps-len(R)-1))

                    if dynamics.is_terminal(s) and not self._terminal_zero:
                        A.append(None)

                    if dynamics.is_terminal(s):
                        break

                    A.append(policy(s, Q, self._episode_policy))

                if reward:
                    rs = reward([(s,a) for s,a in zip(S,A) if s is not None and a is not None])
                    R  = rs[0:len(R)] + [None] * max(len(R) - len(rs), 0)

                SS.append(S)
                AA.append(A)
                RR.append(R)

            times[0] += time.time()-start_time

            start_time = time.time()
            episodes = []
            for S,A,R in zip(SS,AA,RR):

                SAP      = [ (S[0], A[0], self._start_policy) ] + list(zip(S[1:], A[1:], repeat(self._episode_policy)))
                examples = []
                labels   = []

                n_observed_R = len(list([r for r in R if r is not None]))

                if self._n_target is None:
                    starts = list(range(n_observed_R))  #includes the start
                    ends   = [n_observed_R] * len(starts) #doesn't include the end
                else:
                    starts = list(range(n_observed_R-self._n_target+1))
                    ends   = list(range(self._n_target, n_observed_R+1))

                for start,end in zip(starts,ends):

                    if S[start] is None or A[start] is None:
                        continue

                    example = (S[start],A[start])
                    value   = float(sum(R[start:end])) / (end-start)

                    if self._bootstrap > 0 and bootstrap(S, A, Q, end) is not None: 
                        value = value + (self._bootstrap)*(bootstrap(S, A, Q, end)-value)

                    examples.append(example)
                    labels.append(float(value))

                episodes.append((examples, labels, starts, ends, SAP))
            times[1] += time.time()-start_time

            start_time = time.time()
            iter_episodes.append(episodes)

            training_examples = []
            training_labels   = []

            for examples, labels, starts, ends, SAP in chain.from_iterable(iter_episodes[-(self._previous_samples+1):]):

                for example, label, start, end in zip(examples, labels, starts, ends):
                    training_examples.append(example)
                    training_labels.append(label)

            Q = self._regressor.fit(training_examples, training_labels)
            times[2] += time.time()-start_time

        return DirectEstimateIteration.GreedyPolicy(Q)

class StableBaseline(PolicyLearner):
    
    """A facade to provide StableBasline algorithm compatibility.
    https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
    """

    class StableBaselinePolicy(Policy):
        def __init__(self, policy: Union[PPO, A2C]) -> None:
            self._policy = policy

        def __call__(self, state: State, actions: Sequence[Action]) -> Action:
            return self._policy.predict(state, deterministic=True)[0]

    def __init__(self, algorithm: Literal["A2C", "PPO"], n_envs:int, n_steps:int, policy:str="MlpPolicy", **kwargs):

        self._algorithm = algorithm
        self._n_envs    = n_envs
        self._n_steps   = n_steps
        self._policy    = policy
        self._kwargs    = kwargs

    def learn_policy(self, model: GymModel, reward: Reward) -> Policy:

        algorithm   = A2C if self._algorithm == "A2C" else PPO
        environment = make_vec_env(lambda: GymEnvironment(model, reward), n_envs=self._n_envs)
        policy      = algorithm(self._policy, environment, verbose=0, **self._kwargs).learn(self._n_steps)

        return StableBaseline.StableBaselinePolicy(policy)

class KernelProjection(RewardLearner):

    """Kernel extension to Abeel and Ngs Apprenticeship Learning via IRL"""

    class KernelReward(Reward):

        def __init__(self, alpha: 'KernelVector') -> None:
            self._alpha  = alpha
            self._reward = lambda s: sum([ c*self._alpha.kernel([i],[s])[0][0] for c,i in self._alpha ])

        def __call__(self, states_actions: Sequence[Tuple[State,Action]]):       
            return list(map(self._reward,states_actions))

    def __init__(self,
        policy_learner: PolicyLearner,
        kernel        : Kernel,
        gamma         : float,
        epsilon       : float,
        max_iterations: int,
        is_verbose    : bool = False) -> None:

        self._policy_learner = policy_learner
        self._kernel         = kernel
        self._gamma          = gamma
        self._epsilon        = epsilon
        self._is_verbose     = is_verbose
        self._max_iterations = max_iterations

    def learn_reward(self, model: SimModel, expert_episodes: Sequence[Episode]) -> Reward:

        fe_length = min([ len(e.actions) for e in expert_episodes])

        expert_visits = self.expected_visitation_in_kernel_space(expert_episodes)

        alphas          : List[KernelVector] = []
        rewards         : List[Reward      ] = []
        rewards_visits  : List[KernelVector] = []

        print("Running KernelProjectionIRL")

        start             = time.time()
        alpha             = KernelVector(self._kernel, torch.randn((len(expert_visits))).tolist(), expert_visits.items)
        reward            = KernelProjection.KernelReward(alpha)
        reward_policy     = self._policy_learner.learn_policy(model, reward)
        reward_episodes   = [ Episode.generate(model,reward_policy,fe_length) for _ in range(100) ]
        reward_visits     = self.expected_visitation_in_kernel_space(reward_episodes)
        new_convex_visits = reward_visits

        alphas.append(alpha)
        rewards.append(reward)
        rewards_visits.append(reward_visits)

        i = 1
        t = math.sqrt((expert_visits - new_convex_visits) @ (expert_visits - new_convex_visits))
        j = float("inf")

        if self._is_verbose:
            print(f"i={i} t={round(t,4)} j={round(j,4)} time={round(time.time()-start,4)}")

        while t > self._epsilon and j > self._epsilon and i < self._max_iterations:

            old_convex_visits = new_convex_visits

            start           = time.time()
            alpha           = expert_visits - old_convex_visits
            reward          = KernelProjection.KernelReward(alpha)
            reward_policy   = self._policy_learner.learn_policy(model, reward)
            reward_episodes = [ Episode.generate(model,reward_policy,fe_length) for _ in range(100) ]
            reward_visits   = self.expected_visitation_in_kernel_space(reward_episodes)

            alphas.append(alpha)
            rewards.append(reward)
            rewards_visits.append(reward_visits)

            theta_num = (reward_visits - old_convex_visits) @ (expert_visits - old_convex_visits)
            theta_den = (reward_visits - old_convex_visits) @ (reward_visits - old_convex_visits)
            theta     = theta_num/theta_den

            new_convex_visits = old_convex_visits + theta * (reward_visits - old_convex_visits)

            i += 1
            t = (expert_visits - new_convex_visits).norm()
            j = (old_convex_visits - new_convex_visits).norm()

            if self._is_verbose:
                print(f"i={i} t={round(t,4)} j={round(j,4)} time={round(time.time()-start,4)}")

        min_dist   = float("inf")
        min_reward = None
        min_alpha  = None

        for alpha, reward, reward_visits in zip(alphas, rewards, rewards_visits):
            if (reward_visits - expert_visits).norm() < min_dist:
                min_dist   = (reward_visits - expert_visits).norm()
                min_reward = reward
                min_alpha  = alpha

        if min_reward is None or min_alpha is None: raise Exception("No rewards were found")

        return min_reward

    def expected_visitation_in_kernel_space(self, episodes: Sequence[Episode]) -> KernelVector:

        states  = list(chain.from_iterable([zip(e.states,e.actions) for e in episodes]))
        weights = list(chain.from_iterable([self._gamma**torch.arange(len(e.actions)).unsqueeze(1) for e in episodes]))

        #theoretical episode length of length = (# of states / length)
        #print(torch.tensor(weights).T @ self._kernel._featurizer.to_features(states)/len(states))
        return KernelVector(self._kernel, weights, states)/len(states)

class CascadedSupervised(RewardLearner):

    """An implementation of http://www.lifl.fr/~pietquin/pdf/ECML_2013_EKBPMGOP.pdf."""
    ## This implementation could be improved by implementing state/action augmentation 
    ## for the reward learning step as described in the original paper.

    class ProbaClassifier:

        def fit(X,y) -> 'CascadedSupervised.ProbaClassifier':
            ...

        @property
        def classes_(self) -> Sequence[Action]:
            ...

        def predict_proba(self, states: Sequence[State]) -> Sequence[Sequence[float]]:
            ...

    class RegressorReward(Reward):

        def __init__(self, regressor: Regressor) -> None:
            self._sklearn_model = regressor

        def __call__(self, states:Sequence[Tuple[State,Action]] = None) -> Sequence[float]:
            return self._sklearn_model.predict(states) 

    def __init__(self, gamma:float, classifier: ProbaClassifier, regressor: Regressor) -> None:
        self._gamma      = gamma
        self._classifier = classifier
        self._regressor  = regressor

    def learn_reward(self, dynamics: SimModel, expert_episodes: Sequence[Episode]) -> Reward:

        action_classifier = self._action_classifier(expert_episodes)        
        reward_regressor  = self._reward_regressor(expert_episodes, action_classifier)

        return CascadedSupervised.RegressorReward(reward_regressor)

    def _action_classifier(self, expert_episodes: Sequence[Episode]) -> 'CascadedSupervised.ProbaClassifier':

        X = []
        Y = []

        for episode in expert_episodes:

            for state,action in zip(episode.states, episode.actions):

                if isinstance(state, torch.Tensor):
                    state = state.tolist()

                if not isinstance(state, collections.Sequence):
                    state = [state]

                X.append(state)
                Y.append(action)

        return self._classifier.fit(X,Y)

    def _reward_regressor(self, expert_episodes: Sequence[Episode], action_classifier: ProbaClassifier) -> Regressor:

        actions = list(action_classifier.classes_)

        X = []
        Y = []

        for episode in chain(expert_episodes):

            qs = action_classifier.predict_proba([[s] for s in episode.states])

            for i in range(1,len(episode)) :

                previous_Q = qs[i-1,actions.index(episode.actions[i-1])]
                current_Q  = max(qs[i,:])

                X.append([episode.states[i-1], episode.actions[i-1]])
                Y.append(previous_Q-self._gamma*current_Q)

        return self._regressor.fit(X,Y)

class MaxCausalEnt(RewardLearner):

    """https://www.cs.cmu.edu/~kkitani/pdf/HFKB-AAAI15.pdf"""

    class ApproximatePolicy(Policy):

        def __init__(self, actions, Q):
            self._actions = actions
            self._Q       = Q

        def act(self, state) -> Action:
            
            qs = self._Q([(state,action) for action in self._actions])
            Z  = self._soft_max(qs)

            return random.choices(self._actions, weights=np.exp(qs-Z))[0]

        def _soft_max(self, xs: Sequence[float]) -> float:
            max_x = max(xs)
            return max_x + np.log(sum(np.exp(xs-max_x)))

    class ThetaReward(Reward):

        def __init__(self, theta: np.ndarray) -> None:
            self._theta = np.array(theta).squeeze()

        def observe(self, states:Sequence[Tuple[State,Action]] = None) -> Union[Sequence[float], float]:
            return np.array([ s for s,a in states ]) @ self._theta

    def __init__(self, n_ephochs:int=15) -> None:
        self._n_epochs = n_ephochs
        self._skip_n   = 0

    def learn_reward(self, dynamics: SimModel, expert_episodes: Sequence[Episode]) -> Reward: #type:ignore

        r_t0s0a0_features = self._r_featurizer.to_features([(expert_episodes[0].states[0], expert_episodes[0].actions[0])])
        q_t0s0a0_features = self._q_featurizer.to_features([(expert_episodes[0].states[0], expert_episodes[0].actions[0])])

        actions = dynamics.actions()
        r_theta   = torch.rand ( (r_t0s0a0_features.shape[1],1)).double()
        r_theta   = r_theta/r_theta.sum()
        q_theta   = torch.rand (q_t0s0a0_features.shape[1]).double()
        q_theta   = q_theta/q_theta.sum()

        thetas = []
        dists  = []

        training_examples = []

        for episode in expert_episodes:
            for s0,a0,s1 in zip(episode.states[self._skip_n:],episode.actions[self._skip_n:],episode.states[self._skip_n+1:]):
                training_examples.append((s0,a0,s1))

        mu_E = torch.tensor([self._feature_expectation(expert_episodes)])

        for n in range(self._n_epochs):

                # q_theta   = torch.rand (q_t0s0a0_features.shape[1]).double()
                # q_theta   = q_theta/q_theta.sum()

                epoch_examples  = training_examples
                r_s0a0_features = self._r_featurizer.to_features([(s0,a0) for s0,a0,s1 in epoch_examples])
                q_s0a0_features = self._q_featurizer.to_features([(s0,a0) for s0,a0,s1 in epoch_examples])
                q_s1an_features = self._q_featurizer.to_features([(s1,a1) for s0,a0,s1 in epoch_examples for a1 in actions ])

                Rs = r_s0a0_features @ r_theta
                Q  = sklearn.linear_model.SGDRegressor(average=40, warm_start=True)
                X  = q_s0a0_features.tolist()

                random.shuffle(epoch_examples)

                #V1 -- not great

                # for z in range(9):

                #     for i in range(len(epoch_examples)):
                #         q_s0a0_feature = q_s0a0_features[i,:]
                #         q_grad = torch.zeros(len(q_theta))
                #         q_estm = 0

                #         for j in range(len(actions)):
                #             q_s1aj_feature = q_s1an_features[i*len(actions)+j,:]
                #             q_grad += torch.exp((q_s1aj_feature-q_s0a0_feature)@q_theta)*(q_s1aj_feature-q_s0a0_feature)
                #             q_estm += torch.exp((q_s1aj_feature-q_s0a0_feature)@q_theta)

                #         q_theta = q_theta - 1/len(epoch_examples) * (q_estm-torch.exp(-Rs[i])) * q_grad

                #V2

                for i in range(50):

                    q_grad      = 0
                    L           = 0
                    q_theta_bar = q_theta

                    example_indexes = list(range(len(epoch_examples)))
                    random.shuffle(example_indexes)
                    example_indexes = example_indexes[:100]

                    for j in example_indexes:

                        dQ0_dw = q_s0a0_features[j,:]

                        r0  = (r_s0a0_features[j,:]@r_theta).item()
                        Q0  =  q_s0a0_features[j,:]@q_theta
                        Q1s =  q_s1an_features[j*len(actions):(j+1)*len(actions),:]@q_theta

                        dZ_du = self._soft_arg_max(Q1s)
                        du_dw = q_s1an_features[j*len(actions):(j+1)*len(actions),:]

                        ## e = Q0 - self._soft_max(Q1s) - r0
                        ## L = e**2

                        e     = Q0 - (r0 + .9*self._soft_max(Q1s).item())
                        dL_de = e
                        de_dw = dQ0_dw - .9*dZ_du@du_dw
                        dL_dw = dL_de*de_dw

                        L += abs(e)#**2
                        #q_theta = q_theta - 1/len(example_indexes) * dL_dw
                        #q_theta = q_theta*torch.exp(-1/len(example_indexes)*dL_dw)
                        q_grad += dL_dw
                    
                    #print(f"L: {round(L.item()/len(example_indexes),3)}")
                    q_theta = q_theta*torch.exp(-1/len(example_indexes)*q_grad)
                    #q_theta = q_theta - 1/len(example_indexes) * q_grad


                print(f"L{n}: {round(L.item()/len(example_indexes),3)}")

                #V3

                # for i in range(1500):

                #     q_grad      = 0
                #     L           = 0
                #     q_theta_bar = q_theta

                #     example_indexes = list(range(len(epoch_examples)))
                #     random.shuffle(example_indexes)
                #     example_indexes = example_indexes[:50]

                #     for j in example_indexes:

                #         dQ0_dw = q_s0a0_features[j,:]

                #         r0  = (r_s0a0_features[j,:]@r_theta).item()
                #         Q0  =  q_s0a0_features[j,:]@q_theta
                #         Q1s =  q_s1an_features[j*len(actions):(j+1)*len(actions),:]@q_theta_bar

                #         dZ_du = self._soft_arg_max(Q1s)
                #         du_dw = q_s1an_features[j*len(actions):(j+1)*len(actions),:]

                #         ## e = Q0 - self._soft_max(Q1s) - r0
                #         ## L = e**2

                #         e     = Q0 - (r0 + .9*self._soft_max(Q1s).item())
                #         dL_de = e
                #         de_dw = dQ0_dw #- .9*dZ_du@du_dw
                #         dL_dw = dL_de*de_dw

                #         L += abs(e)#**2
                #         q_theta = q_theta - 1/len(example_indexes) * dL_dw
                #         #q_theta = q_theta*torch.exp(-1/len(example_indexes)*dL_dw)
                #         #q_grad += dL_dw
                    
                #     #print(f"L: {round(L.item()/len(example_indexes),3)}")
                #     #q_theta = q_theta*torch.exp(-1/len(example_indexes)*q_grad)
                #     #q_theta = q_theta - 1/len(example_indexes) * q_grad

                # print(f"L{n}: {round(L.item()/len(example_indexes),3)}")

                #V4 -- pretty good

                # for i in range(45):

                #     if i==0:
                #         Qs = Rs
                #     else:
                #         Qs = Rs + self._soft_max(Q.predict(q_s1an_features).reshape((len(epoch_examples), len(actions))))
                    
                #     Q.fit(X, Qs.squeeze().tolist())

                #policy_Q        = lambda sa: Q.predict(self._q_featurizer.to_features(sa))
                policy_Q        = lambda sa: self._q_featurizer.to_features(sa)@q_theta
                policy_A        = dynamics.actions()
                policy          = MaxCausalEnt.ApproximatePolicy(policy_A, policy_Q)
                policy_episodes = [ dynamics.make_episode(policy, 20) for _ in range(100)]

                t = n+1
                l = 1

                #standard stochastic approximation
                mu_P     = torch.tensor([self._feature_expectation(policy_episodes)])
                gradient = (mu_E-mu_P).T
                r_theta  = r_theta + (l/t)*gradient
                #theta    = theta*torch.exp((l/t)*gradient)

                #exponential gradient approximation
                # for policy_episode in policy_episodes:
                #    mu_P     = torch.tensor([self._feature_expectation([policy_episode])])
                #    gradient = (mu_E-mu_P).T
                #    theta    = theta*torch.exp((l/t)*gradient)
                #    t+=1

                total_grad = (mu_E-torch.tensor([self._feature_expectation(policy_episodes)]))

                thetas.append(r_theta.squeeze().tolist())
                dists.append(float(total_grad.norm()))

                print( f"R: {round(float(total_grad.norm()),5):6.10f}")

                if total_grad.norm() < 0.25:
                    break

        best_theta = thetas[dists.index(min(dists))]

        print( f"{round(min(dists),5):6.10f}")

        return MaxCausalEnt.ThetaReward(thetas[-1], self._r_featurizer), thetas[-1], self._feature_expectation(expert_episodes)

    def _feature_expectation(self, expert_episodes):

        expert_feature_sum = None
        expert_state_count = None

        for episode in expert_episodes:

            if len(episode.states) <= self._skip_n or len(episode.actions) <= self._skip_n: continue

            featureized_states = self._r_featurizer.to_features(list(zip(episode.states[self._skip_n:], episode.actions[self._skip_n:])))

            if expert_feature_sum is None:
                expert_state_count = len(featureized_states)
                expert_feature_sum = featureized_states.sum(dim=0)
            else:
                expert_state_count += len(featureized_states)
                expert_feature_sum += featureized_states.sum(dim=0)

        return (expert_feature_sum / expert_state_count).tolist()

    def _soft_max(self, xs: np.ndarray) -> float:
        xs = np.array(xs)

        if len(xs.shape) == 1:
            xs = np.expand_dims(xs,axis=0)

        max_xs = xs.max(1)[:,np.newaxis]
        return max_xs + np.log(np.exp(xs-max_xs).sum(1))[:,np.newaxis]

    def _soft_arg_max(self, xs: np.ndarray) -> np.ndarray:
        xs = torch.exp(xs-max(xs))
        return torch.exp(xs) / torch.exp(xs).sum()
