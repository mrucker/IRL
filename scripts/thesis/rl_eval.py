import statistics
import time
import math
import random

import torch
import torch.nn

from gym.spaces import Box

from stable_baselines3 import PPO, DQN, A2C

from sklearn.tree import DecisionTreeRegressor #type: ignore
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor #type:ignore
from sklearn.neural_network import MLPRegressor

from matplotlib import rcParams #type: ignore
import matplotlib.pyplot as plt #type: ignore

from combat.representations import SA_Featurizer, CachedFeaturizer
from combat.environments import GymEnvironment, RewardEnvironment
from combat.algorithms import DirectEstimateIteration, StableBaseline, SklearnRegressor

from combat.domains import thesis

# plot_title = "Acrobot-v1"
# dynamics = GymEnvironment('Acrobot-v1')
# eval_reward_count   = 5
# eval_episode_length = 500
# eval_episode_count  = 50
# def make_random_reward(): return None
# dei_hyperparams = [6, 75, 500, None, SklearnRegressor(MLPRegressor())]
# dqn_hyperparams = {"learning_rate": 6.3*10**-4, "batch_size":128, "buffer_size":50000, "learning_starts":0, "gamma":0.99, "target_update_interval": 250, "train_freq": 4, "gradient_steps":-1, "exploration_fraction":0.12, "exploration_final_eps":0.1, "policy_kwargs":{"net_arch":[256,256]} }
# ppo_hyperparams = {'batch_size': 128, 'n_steps': 64, 'gamma': 0.99, 'learning_rate': 0.03867310281066357, 'ent_coef': 0.013511290785945073, 'clip_range': 0.1, 'n_epochs': 5, 'gae_lambda': 1.0, 'max_grad_norm': 0.3, 'vf_coef': 0.055200240847359416, "policy_kwargs": {'net_arch': [dict(pi=[64, 64], vf=[64, 64])], 'activation_fn': torch.nn.Tanh }}
# a2c_hyperparams = {'gamma': 0.95, 'normalize_advantage': True, 'max_grad_norm': 2, 'use_rms_prop': True, 'gae_lambda': 0.9, 'n_steps': 32, 'learning_rate': 0.004584759419514452, 'ent_coef': 1.8970768683821307e-05, 'vf_coef': 0.6548027338540534, "policy_kwargs":{'ortho_init': False, 'net_arch': [dict(pi=[64, 64], vf=[64, 64])], 'activation_fn': torch.nn.ReLU} }

# plot_title = "CartPole-v1"
# dynamics = GymEnvironment('CartPole-v1')
# eval_reward_count   = 5
# eval_episode_length = 500
# eval_episode_count  = 50
# def make_random_reward(): return None
# dei_args, dei_kwargs = [8, 75, 300, 250, SklearnRegressor(MLPRegressor(max_iter=300))], dict(terminal_zero=True)
# dqn_hyperparams = {"learning_rate": 2.3*10**-3, "batch_size":64, "buffer_size":100000, "learning_starts":1000, "gamma":0.99, "target_update_interval": 10, "train_freq": 256, "gradient_steps":128, "exploration_fraction":0.16, "exploration_final_eps":0.04, "policy_kwargs":{"net_arch":[256,256]} }
# ppo_hyperparams = {"n_steps": 16, "gae_lambda":0.98, "gamma":0.99, "n_epochs": 4, "ent_coef":0.}
# a2c_hyperparams = {"ent_coef":0.}

# plot_title = "MountainCar-v0"
# dynamics = GymEnvironment('MountainCar-v0')
# eval_reward_count   = 1
# eval_episode_length = 500
# eval_episode_count  = 50
# def make_random_reward(): return None
# dei_args, dei_kwargs = [8, 75, 300, 250, SklearnRegressor(MLPRegressor(max_iter=300))], dict(terminal_zero=True)
# dqn_hyperparams = {}
# ppo_hyperparams = {}
# a2c_hyperparams = {}

plot_title = "LunarLander-v2"
dynamics = GymEnvironment('LunarLander-v2')
eval_reward_count   = 1
eval_episode_length = 500
eval_episode_count  = 50
def make_random_reward(): return None
dei_args, dei_kwargs = [4, 100, 300, None, SklearnRegressor(MLPRegressor(hidden_layer_sizes=(64,64), max_iter=1000))], dict(terminal_zero=False)
dqn_hyperparams      = {}
ppo_hyperparams      = {}
a2c_hyperparams      = {}

# plot_title = "Experiment Game"
# dynamics = thesis.ThesisDynamics()
# eval_reward_count   = 2
# eval_episode_length = 40
# eval_episode_count  = 200
# def make_random_reward(): return thesis.RandomReward()
# observer = (Box(low=-float('inf'), high=float('inf'), shape=(9,)), thesis.ValueFeatures())
# dei_hyperparams = [20, 30, 25, 8, SklearnRegressor(AdaBoostRegressor()), SA_Featurizer(CachedFeaturizer(thesis.ValueFeatures()))]
# dqn_hyperparams = {'observer': observer}
# ppo_hyperparams = {'observer': observer, 'batch_size': 128, 'n_steps': 64, 'gamma': 0.99, 'learning_rate': 0.03867310281066357, 'ent_coef': 0.013511290785945073, 'clip_range': 0.1, 'n_epochs': 5, 'gae_lambda': 1.0, 'max_grad_norm': 0.3, 'vf_coef': 0.055200240847359416, "policy_kwargs": {'net_arch': [dict(pi=[64, 64], vf=[64, 64])], 'activation_fn': torch.nn.Tanh }}
# a2c_hyperparams = {'observer': observer, "policy_kwargs": {"ortho_init":True, "net_arch":[dict(pi=[256, 256], vf=[256, 256])], "activation_fn":torch.nn.Tanh },"normalize_advantage":False, "vf_coef":0.40636068373067824, "ent_coef":8.270807139587433e-08, "gae_lambda":0.92, "learning_rate":0.012591031680048069, "n_steps": 512, "max_grad_norm":0.9, "gamma":0.999, "use_rms_prop": False}

learners = [

    # ["Direct Iteration greed uniform (0.0)" , DirectEstimateIteration(*dei_args, **dei_kwargs, importance_sample=False, episode_policy='greedy', bootstrap=0)],

    # ["Direct Iteration greed weighted (0.0)", DirectEstimateIteration(*dei_args, **dei_kwargs, importance_sample=True, episode_policy='greedy', bootstrap=0)],

    # ["Direct Iteration greed uniform (0.05)" , DirectEstimateIteration(*dei_args, **dei_kwargs, importance_sample=False, episode_policy='greedy', bootstrap=0.05)],

    # ["Direct Iteration greed weighted (0.05)", DirectEstimateIteration(*dei_args, **dei_kwargs, importance_sample=True, episode_policy='greedy', bootstrap=0.05)],

    # ["Direct Iteration soft uniform (0.0)"  , DirectEstimateIteration(*dei_args, **dei_kwargs, importance_sample=False, episode_policy='softmax', bootstrap=0)],

    # ["Direct Iteration soft weighted (0.0)" , DirectEstimateIteration(*dei_args, **dei_kwargs, importance_sample=True, episode_policy='softmax', bootstrap=0)],

    # ["Direct Iteration soft uniform (0.05)"  , DirectEstimateIteration(*dei_args, **dei_kwargs, importance_sample=False, episode_policy='softmax', bootstrap=.05)],

    # ["Direct Iteration soft weighted (0.05)" , DirectEstimateIteration(*dei_args, **dei_kwargs, importance_sample=True, episode_policy='softmax', bootstrap=.05)],

    #["Deep Q-Network", StableBaseline(DQN, 15000, **dqn_hyperparams)],

    ["Proximal Policy Opt", StableBaseline(PPO, 180000, **ppo_hyperparams)],

    #["Advantage Actor-Critic 1", StableBaseline(A2C, 15000, **a2c_hyperparams)],
]

box_values = []
box_labels = []

rewards = [ make_random_reward() for _ in range(eval_reward_count) ]
#rewards = [ make_random_reward() ] * eval_reward_count

print("")
for i, (desc, learner) in enumerate(learners):

    values = []
    times  = []

    box_labels.append(desc)
    box_values.append([])

    for j, reward in enumerate(rewards):

        start    = time.time()
        policy   = learner.learn_policy(dynamics, reward)
        end      = time.time()

        eval_environment = RewardEnvironment(dynamics, reward)
        eval_environment.seed(200)
        episodes = [eval_environment.make_episode(policy, eval_episode_length) for _ in range(eval_episode_count)]

        values.append(statistics.mean([ sum(e.rewards) for e in episodes ]))
        times .append(end-start)

        print(values[-1])

        box_values[-1].append(values[-1])

    E_value  = statistics.mean(values)
    M_value  = statistics.median(values)
    SE_value = statistics.stdev(values)/math.sqrt(len(rewards)) if len(rewards)  > 1 else 0
    E_time   = statistics.mean(times)
    SE_time  = statistics.stdev(times)/math.sqrt(len(rewards)) if len(rewards)  > 1 else 0

    print(f"{round(E_value,2):<06}±{round(SE_value,2):<06}, {round(M_value,2):<06}, {round(E_time,2):<04}±{round(SE_time,2):<04}, {desc}")

rcParams.update({'figure.autolayout': True})

print(box_values)
print(box_labels)

box_values.reverse()
box_labels.reverse()

fig, ax = plt.subplots()
ax.set_title(plot_title)

bp = ax.boxplot(box_values, labels=box_labels, vert=False, showmeans=True, medianprops=dict(linestyle='dotted', color='dimgray'), meanprops=dict(markerfacecolor='dimgray',markeredgecolor='dimgray'))

ax.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'])
ax.tick_params(axis='y', labelsize='small')

plt.show()