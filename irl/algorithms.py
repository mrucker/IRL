import copy
import collections
import random
import time
import math

from functools import reduce
from abc import ABC, abstractmethod
from itertools import chain, count, repeat
from typing import Sequence, List, Callable, Any, Tuple, Optional, Literal, Union, Dict

import numpy as np
from numpy.core.records import get_remaining_size
import torch
import gym.spaces

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

from combat.representations import (
    Kernel, Action, Observation, Policy, Reward, Regressor, 
    SA_Featurizer, Episode, KernelVector, KernelVectorReward, 
    IdentityFeaturizer
)

from combat.environments import GymEnvironment, RewardEnvironment, SimEnvironment, MassEnvironment
from combat.representations import Featurizer
from combat.visualizations import VideoPlot
from combat.domains import phase3

class PolicyLearner(ABC):
    @abstractmethod
    def learn_policy(self, dynamics, reward: Reward) -> Policy:
        ...

class RewardLearner(ABC):
    @abstractmethod
    def learn_reward(self, dynamics, episodes: Sequence[Episode]) -> Reward:
        ...

class RegressionLearner(ABC):

    @abstractmethod
    def learn_regression(self, X, Y) -> Regressor:
        ...

class SklearnRegressor(RegressionLearner):

    @staticmethod
    def _sklearn_safe(X):
        if isinstance(X, torch.Tensor):
            return X.tolist()

        if isinstance(X, collections.Sequence) and isinstance(X[0], torch.Tensor):
            return [ x.tolist() for x in X]

        return X

    class SklearnRegressor(Regressor):
        def __init__(self, sklearn_regressor) -> None:
            self._sklearn_regressor = sklearn_regressor

        def predict(self, X) -> torch.Tensor:

            X = SklearnRegressor._sklearn_safe(X)

            return torch.Tensor(self._sklearn_regressor.predict(X))

    def __init__(self, sklearn_learner):
        self._learner = sklearn_learner

    def learn_regression(self, X, Y, sample_weight = None) -> Callable[[Any], torch.Tensor]:

        learner = copy.copy(self._learner)

        X = SklearnRegressor._sklearn_safe(X)
        Y = SklearnRegressor._sklearn_safe(Y)

        is_weightable = not isinstance(learner, sklearn.neural_network.MLPRegressor)

        if sample_weight is not None and is_weightable:
            return SklearnRegressor.SklearnRegressor(learner.fit(X,Y, sample_weight=sample_weight))
        else:
            return SklearnRegressor.SklearnRegressor(learner.fit(X,Y))

class ValueIteration(PolicyLearner):

    class GreedyLookupPolicy(Policy):
        def __init__(self, Q: torch.Tensor, state_indexes: Dict[Any,int], actions: Sequence[Action]):
            self._Q           = Q
            self._state_index = state_indexes
            self._actions     = actions

        def act(self, state: Any) -> Action:

            state = tuple(state.tolist()) if isinstance(state, torch.Tensor) else state

            s_i = self._state_index[state]
            a_i = int(torch.argmax(self._Q[:,s_i,:]))

            return self._actions[a_i]

    def __init__(self, gamma: float=0.9, epsilon: float=0.1, iterations: int = None) -> None:
        
        assert epsilon < 1, "if epsilon is >= 1 no learning will occur"

        self._gamma      = gamma
        self._epsilon    = epsilon
        self._iterations = iterations or float('inf')

    def learn_policy(self, dynamics: MassEnvironment, reward: Reward) -> Policy:
        S = dynamics.S
        A = dynamics.A
        R = torch.full((len(A),len(S),1), 0).float()
        Q = torch.full((len(A),len(S),1), 0).float()
        T = dynamics.transition_tensor

        for a_i in range(len(A)):
            R[a_i,:] = torch.tensor(reward.observe(states=list(zip(S,repeat(A[a_i]))))).unsqueeze(1)

        S_index = dict(zip(map(tuple,S.tolist()),count())) if isinstance(S, torch.Tensor) else dict(zip(S,count())) #type: ignore

        assert R.shape[0] == len(A)
        assert R.shape[1] == len(S)
        #assert R.shape[1] == 1, "R is expected to be a column vector"

        #scale all rewards to length of 1 so that epsilon is meaningful regardless
        #of the given rewards. This scaling has no effect on the optimal policy.
        
        min = torch.min(R)
        max = torch.max(R)
        R   = (R-min) / (max-min)

        old_V, new_V = torch.zeros((len(S),1)), torch.max(R, 0, keepdim=False)[0]

        iteration = 0

        while torch.max(torch.abs(new_V-old_V)) > self._epsilon and iteration < self._iterations:
            iteration += 1
            Q[:] = R[:] + torch.bmm(T, (self._gamma * new_V).unsqueeze(0).repeat(len(A),1,1))

            #This variation seems to run a tiny bit faster than the torch.bmm method above
            #however its CPU usage is much much higher. Probably more than it is worth. There
            #doesn't appear to be any real change in memory usage between the two approaches.
            # V = (self._gamma * new_V)
            # for a_i in range(len(A)):
            #     Q[a_i,:] = R[a_i,:] + T[a_i,] @ V

            old_V, new_V = new_V, torch.max(Q, 0, keepdim=False)[0]

        return ValueIteration.GreedyLookupPolicy(Q, S_index, A)

class DirectEstimateIteration(PolicyLearner):

    # Add importance weighting
    # Add better bootstrap support
    # Add first visit only 

    class NullQ(Regressor):

        def predict(self, X) -> torch.Tensor:
            return torch.tensor([0]* len(X)).float()

    class GreedyRegressionPolicy:
        def __init__(self, Q: Regressor, featurizer: Featurizer, actions: Sequence[Action]) -> None:
            self._Q = Q
            self._featurizer = featurizer
            self._actions = actions

        def act(self, state) -> Action:
            qs = self._Q.predict(self._featurizer.to_features(list(zip(repeat(state), self._actions))))
            max_indexes = (qs == qs.max()).int().nonzero(as_tuple=True)[0].tolist()
            return self._actions[random.choice(max_indexes)]

    def __init__(self, 
        n_iterations: int,
        n_episodes: int,
        n_steps: int,
        n_target: Optional[int],
        regression_learner: RegressionLearner,
        sa_featurizer: Featurizer = SA_Featurizer(),
        bootstrap: float = 0,
        start_policy: Union[Literal['softmax'], Literal['greedy'], Literal['epsilon']] = 'softmax',
        episode_policy: Union[Literal['softmax'], Literal['greedy'], Literal['epsilon']] = 'greedy',
        previous_samples: Optional[int] = None,
        terminal_zero: bool = True,
        importance_sample: bool = False):

        self._regression   = regression_learner
        self._featurizer   = sa_featurizer
        self._n_iterations = n_iterations
        self._n_steps      = n_steps
        self._n_episodes   = n_episodes
        self._n_target     = n_target

        self._bootstrap = bootstrap
        self._start_policy = start_policy
        self._episode_policy = episode_policy
        self._previous_samples = previous_samples if previous_samples is not None else n_iterations
        self._terminal_zero = terminal_zero
        self._importance_sample = importance_sample

    def learn_policy(self, dynamics: SimEnvironment, reward: Reward) -> Policy:

        times = [0.,0.,0.]

        def q_domain(state, actions):
            return self._featurizer.to_features(list(zip(repeat(state), actions)))

        def policy(s0, Q, policy_type: Union[Literal['softmax'], Literal['greedy'], Literal['epsilon']]):
            
            a0s = dynamics.actions()

            if policy_type == 'epsilon':
                return random.choice(a0s) if random.random() < 0.5 else policy(s0, Q, 'greedy')

            qs = Q.predict(q_domain(s0, a0s))

            if policy_type == 'softmax':
                es = torch.exp(qs-qs.max())
                ps = es/sum(es)
                return random.choices(a0s, ps, k=1)[0]

            if policy_type == 'greedy':
                max_indexes = (qs == qs.max()).int().nonzero(as_tuple=True)[0].tolist()
                return a0s[random.choice(max_indexes)]

            raise Exception(f"Unrecognized policy: {policy_type}")

        def bootstrap(S,A,Q,end):
            if end >= len(A) or S[end] is None or A[end] is None:
                return None
            else:
                return Q.predict(q_domain(S[end],[A[end]]))[0].item()

        def probabilities(trajectory,Q):
            
            actions = dynamics.actions()

            probabilities = []

            for state,action,policy_type in trajectory:

                if action is None:
                    break

                if policy_type == 'epsilon':
                    epsilon, all_uniform = 0.5, 1/len(actions)
                    p = epsilon*all_uniform + probabilities([(state,action,'greedy')],Q)

                if policy_type == 'softmax':
                    qs = Q.predict(q_domain(state, actions))
                    es = torch.exp(qs-qs.max())
                    ps = es/sum(es)
                    p = float(ps[actions.index(action)])

                if policy_type == 'greedy':
                    qs = Q.predict(q_domain(state, actions))
                    max_indexes = (qs == qs.max()).int().nonzero(as_tuple=True)[0].tolist()
                    p = int(actions.index(action) in max_indexes)/len(max_indexes)
                    
                probabilities.append(p)

            return probabilities

        Q = DirectEstimateIteration.NullQ()

        iter_episodes: Tuple[List[Observation], List[float], Sequence[Tuple[Observation,Action,str]], Sequence[float]] = []

        for iteration in range(self._n_iterations):
            SS: List[List[Observation]] = []
            AA: List[List[Action]]      = []
            RR: List[Sequence[float]]   = []

            start_time = time.time()
            for _ in range(self._n_episodes):

                S: List[Observation] = []
                A: List[Action     ] = []
                R: List[float      ] = []

                S.append(dynamics.reset())
                A.append(policy(S[-1], Q, self._start_policy))

                for _ in range(1, self._n_steps):

                    s,r,terminal = dynamics.step(A[-1])[0:3]

                    R.append(r)
                    S.append(s)

                    if terminal and self._terminal_zero:
                        A.extend([None]*(self._n_steps-len(A)  ))
                        S.extend([None]*(self._n_steps-len(S)  ))
                        R.extend([0]   *(self._n_steps-len(R)-1))

                    if terminal and not self._terminal_zero:
                        A.append(None)

                    if terminal:
                        break

                    A.append(policy(s, Q, self._episode_policy))

                if reward:
                    rs = list(reward.observe(states=[s for s,a in zip(S,A) if s is not None and a is not None]))
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

                n_observed_R = len(list(filter(None,R)))

                if self._n_target is None:
                    starts = list(range(n_observed_R))  #includes the start
                    ends   = [n_observed_R] * len(starts) #doesn't include the end
                else:
                    starts = list(range(n_observed_R-self._n_target+1))
                    ends   = list(range(self._n_target, n_observed_R+1))

                for start,end in zip(starts,ends):

                    if S[start] is None or A[start] is None:
                        continue

                    example = q_domain(S[start],[A[start]])[0]
                    value  = float(sum(R[start:end])) / (end-start)

                    if self._bootstrap > 0 and bootstrap(S, A, Q, end) is not None: 
                        value = value + (self._bootstrap)*(bootstrap(S, A, Q, end)-value)

                    examples.append(example)
                    labels.append(float(value))

                probs = probabilities(SAP,Q) if self._importance_sample else []
                episodes.append((examples, labels, starts, ends, SAP, probs))
            times[1] += time.time()-start_time

            start_time = time.time()
            iter_episodes.append(episodes)

            training_examples = []
            training_labels   = []
            training_weights  = [] if self._importance_sample else None

            for examples, labels, starts, ends, SAP, old_probs in chain.from_iterable(iter_episodes[-(self._previous_samples+1):]):

                prod = lambda xs: reduce(lambda x,y: x*y, xs)
                
                if self._importance_sample:
                    new_probs = probabilities(SAP,Q)

                for example, label, start, end in zip(examples, labels, starts, ends):
                    training_examples.append(example)
                    training_labels.append(label)

                    if self._importance_sample:
                        p = prod(new_probs[start:end])/prod(old_probs[start:end])
                        training_weights.append(p)

            Q = self._regression.learn_regression(training_examples, training_labels, training_weights)
            times[2] += time.time()-start_time

            #print(times)

        return DirectEstimateIteration.GreedyRegressionPolicy(Q, self._featurizer, dynamics.actions())

class StableBaseline(PolicyLearner):
    
    class StableBaselinePolicy:
        def __init__(self, policy, observer: Featurizer = None) -> None:
            self._policy = policy
            self._observer = observer

        def act(self, s) -> Action:

            if self._observer:
                return self._policy.predict(self._observer.to_features([s])[0], deterministic=True)[0]
            else:
                return self._policy.predict(s, deterministic=True)[0]

    def __init__(self, 
        algorithm, 
        steps: int, 
        policy:str="MlpPolicy", 
        observer: Tuple[gym.spaces.Space,Featurizer] = None, **kwargs):

        self._alg      = algorithm
        self._stp      = steps
        self._policy   = policy
        self._observer = observer
        self._kwargs   = kwargs

    def learn_policy(self, dynamics: SimEnvironment, reward: Reward) -> Policy:

        if reward:
            dynamics = RewardEnvironment(dynamics, reward)

        if self._observer:
            dynamics = GymEnvironment(dynamics, self._observer[0], self._observer[1])

        policy = self._alg(self._policy, dynamics, verbose=0, **self._kwargs).learn(self._stp)

        if self._observer:
            return StableBaseline.StableBaselinePolicy(policy, self._observer[1])
        else:
            return StableBaseline.StableBaselinePolicy(policy)

class KernelProjection(RewardLearner):

    def __init__(self,
        policy_learner: PolicyLearner,
        kernel        : Kernel,
        gamma         : float,
        epsilon       : float,
        max_iterations: int,
        is_verbose    : bool = True) -> None:

        self._policy_learner = policy_learner
        self._kernel         = kernel
        self._gamma          = gamma
        self._epsilon        = epsilon
        self._is_verbose     = is_verbose
        self._max_iterations = max_iterations

    def learn_reward(self, dynamics: SimEnvironment, expert_episodes: Sequence[Episode]) -> Tuple[Reward, KernelVector, KernelVector]: #type:ignore

        fe_length = min([ len(e.actions) for e in expert_episodes])

        expert_visits = self.expected_visitation_in_kernel_space(expert_episodes)

        alphas          : List[KernelVector] = []
        rewards         : List[Reward      ] = []
        rewards_visits  : List[KernelVector] = []

        print("Running KernelProjectionIRL")

        start             = time.time()
        alpha             = KernelVector(self._kernel, torch.randn((len(expert_visits))).tolist(), expert_visits.items)
        reward            = KernelVectorReward(alpha)
        reward_policy     = self._policy_learner.learn_policy(dynamics, reward)
        reward_episodes   = [ dynamics.make_episode(reward_policy,fe_length) for _ in range(100) ]
        reward_visits     = self.expected_visitation_in_kernel_space(reward_episodes)
        new_convex_visits = reward_visits

        alphas.append(alpha)
        rewards.append(reward)
        rewards_visits.append(reward_visits)

        i = 1
        t = (expert_visits - new_convex_visits).norm()
        j = float("inf")

        if self._is_verbose:
            print(f"i={i} t={round(t,4)} j={round(j,4)} time={round(time.time()-start,4)}")

        while t > self._epsilon and j > self._epsilon and i < self._max_iterations:

            old_convex_visits = new_convex_visits

            start           = time.time()
            alpha           = expert_visits - old_convex_visits
            reward          = KernelVectorReward(alpha)
            reward_policy   = self._policy_learner.learn_policy(dynamics, reward)
            reward_episodes = [ dynamics.make_episode(reward_policy,fe_length) for _ in range(100) ]
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
        min_visits = None

        for alpha, reward, reward_visits in zip(alphas, rewards, rewards_visits):
            if (reward_visits - expert_visits).norm() < min_dist:
                min_dist   = (reward_visits - expert_visits).norm()
                min_reward = reward
                min_alpha  = alpha
                min_visits = reward_visits

        if min_reward is None or min_alpha is None: raise Exception("No rewards were found")

        return min_reward, min_alpha, expert_visits, min_visits

    def expected_visitation_in_kernel_space(self, episodes: Sequence[Episode]) -> KernelVector:

        states  = list(chain.from_iterable([zip(e.states,e.actions) for e in episodes]))
        weights = list(chain.from_iterable([self._gamma**torch.arange(len(e.actions)).unsqueeze(1) for e in episodes]))

        #theoretical episode length of length = (# of states / length)
        #print(torch.tensor(weights).T @ self._kernel._featurizer.to_features(states)/len(states))
        return KernelVector(self._kernel, weights, states)/len(states)

class CascadedSupervised(RewardLearner):

    class SklearnReward(Reward):

        def __init__(self, sklearn_model, featurizer: Featurizer) -> None:
            self._sklearn_model = sklearn_model
            self._featurizer    = featurizer

        def observe(self, state: Observation = None, states:Sequence[Observation] = None) -> Union[Sequence[float], float]:
            
            if state is not None: 
                state = self._featurizer.to_features([state])

            if states is not None:
                states = self._featurizer.to_features(states)

            if isinstance(state, torch.Tensor):
                state = state.tolist()

            if isinstance(states, torch.Tensor):
                states = states.tolist()

            if state is not None and not isinstance(state, collections.Sequence):
                state = [[state]]

            if states is not None and not isinstance(states[0], collections.Sequence):
                states = [[state] for state in states]

            return self._sklearn_model.predict(state)[0] if states is None else self._sklearn_model.predict(states).tolist()

    class SklearnPolicy(Policy):

        def __init__(self, score_based_classifier, actions, featurizer: Featurizer) -> None:
            self._model      = score_based_classifier
            self._featurizer = featurizer
            self._actions    = actions

        def act(self, observation) -> Action:
            scores = torch.tensor([self._model(self._featurizer.to_features([observation])[0], a) for a in self._actions])

            return self._actions[scores.argmax()]

    class RandomPolicy(Policy):
        def __init__(self, actions) -> None:
            self._actions = actions

        def act(self, observation) -> Action:
            return random.choice(self._actions)

    def __init__(self, featurizer: Featurizer = IdentityFeaturizer()) -> None:
        self._featurizer = featurizer

    """An implementation of http://www.lifl.fr/~pietquin/pdf/ECML_2013_EKBPMGOP.pdf."""

    def learn_reward(self, dynamics: SimEnvironment, expert_episodes: Sequence[Episode]) -> Reward: #type:ignore

        actions = dynamics.actions()

        aug = [ dynamics.make_episode(CascadedSupervised.RandomPolicy(actions)) for _ in range(int(len(expert_episodes)/2))]
        aug = []

        score_based_classifier      = self._rf_score_based_classifier(expert_episodes, actions)        
        learned_reward, reward_coef = self._sgd_reward_function(expert_episodes+aug, actions, score_based_classifier)
        expert_feature_expectation  = self._expert_feature_expectation(expert_episodes)

        return learned_reward, reward_coef, expert_feature_expectation

    def _sgd_score_based_classifier(self, expert_episodes, actions):

        model = sklearn.linear_model.SGDClassifier(loss='log')

        for episode in expert_episodes:

            featureized_states = self._featurizer.to_features(episode.states)

            for state,action in zip(featureized_states, episode.actions):

                if isinstance(state, torch.Tensor):
                    state = state.tolist()

                if not isinstance(state, collections.Sequence):
                    state = [state]

                if not isinstance(action, collections.Sequence):
                    action = [action]

                model.partial_fit([state], action, classes=actions)

        return lambda x,a: model.predict_proba([x])[0,actions.index(a)]

    def _rf_score_based_classifier(self, expert_episodes, actions):

        X = []
        Y = []

        for episode in expert_episodes:

            featureized_states = self._featurizer.to_features(episode.states)

            for state,action in zip(featureized_states, episode.actions):
                
                if isinstance(state, torch.Tensor):
                    state = state.tolist()

                if not isinstance(state, collections.Sequence):
                    state = [state]

                X.append(state)
                Y.append(action)

        model = sklearn.ensemble.RandomForestClassifier(n_estimators=300,max_depth=2).fit(X,Y)
        return lambda x,a,m=model: 0 if a not in m.classes_ else m.predict_proba([x])[0,m.classes_.tolist().index(a)]

    def _sgd_reward_function(self, expert_episodes, actions, score_based_classifier):
        reward_regressor = sklearn.linear_model.SGDRegressor()

        pi_c = CascadedSupervised.SklearnPolicy(score_based_classifier, actions, IdentityFeaturizer())

        for episode in chain(expert_episodes):
            
            previous_state = None
            previous_action = None

            regression_X = []
            regression_Y = []

            featurized_states = self._featurizer.to_features(episode.states)

            for state,action in zip(featurized_states, episode.actions):
                
                if isinstance(state, torch.Tensor):
                    state = state.tolist()

                if not isinstance(state, collections.Sequence):
                    state = [state]

                if previous_state is not None:

                    previous_Q = score_based_classifier(previous_state,previous_action)
                    current_Q  = score_based_classifier(state, pi_c.act(state))

                    regression_X.append(previous_state)
                    regression_Y.append(previous_Q-current_Q)

                    reward_regressor.partial_fit([previous_state], [previous_Q-current_Q])

                previous_state, previous_action = state, action

        return CascadedSupervised.SklearnReward(reward_regressor, self._featurizer), reward_regressor.coef_.tolist()

    def _expert_feature_expectation(self, expert_episodes):

        expert_feature_sum = None
        expert_state_count = None

        for episode in expert_episodes:
            
            featureized_states = self._featurizer.to_features(episode.states)

            if expert_feature_sum is None:
                expert_state_count = len(featureized_states)
                expert_feature_sum = featureized_states.sum(dim=0)
            else:
                expert_state_count += len(featureized_states)
                expert_feature_sum += featureized_states.sum(dim=0)
        
        return (expert_feature_sum / expert_state_count).tolist()

class MaxCausalEnt(RewardLearner):

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

        def __init__(self, theta: np.ndarray, r_featurizer: Featurizer) -> None:
            self._theta        = np.array(theta).squeeze()
            self._r_featurizer = r_featurizer

        def observe(self, state: Observation = None, states:Sequence[Observation] = None) -> Union[Sequence[float], float]:
            
            if state is not None: 
                state = self._r_featurizer.to_features([state])

            if states is not None:
                states = self._r_featurizer.to_features(states)

            if isinstance(state, torch.Tensor):
                state = state.tolist()

            if isinstance(states, torch.Tensor):
                states = states.tolist()

            if state is not None and not isinstance(state, collections.Sequence):
                state = [[state]]

            if states is not None and not isinstance(states[0], collections.Sequence):
                states = [[state] for state in states]

            return np.array(state) @ self._theta if states is None else np.array(states) @ self._theta

    def __init__(self, r_featurizer: Featurizer, q_featurizer: Featurizer, n_ephochs:int=15) -> None:
        self._r_featurizer = r_featurizer
        self._q_featurizer = q_featurizer
        self._n_epochs     = n_ephochs

        self._skip_n = 0

    def learn_reward(self, dynamics: SimEnvironment, expert_episodes: Sequence[Episode]) -> Reward: #type:ignore

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