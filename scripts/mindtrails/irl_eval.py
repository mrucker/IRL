import os
import torch

#this is taken from https://github.com/xianyi/OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from itertools import count
from typing import Sequence

import pandas as pd
from sklearn.tree import DecisionTreeRegressor 
from sklearn.pipeline import make_pipeline

from irl.algorithms import DirectEstimateIteration, KernelProjection
from irl.models import Episode
from irl.features import PostStateFeatures
from irl.domains import mindtrails

def make_episodes(df: pd.DataFrame) -> Sequence[Episode]:
    episodes = []

    for session_name in df["session_name"].unique():
        sdf = df[df["session_name"] == session_name]

        states =[]
        actions=[]
    
        for i, row in enumerate(sdf.sort_values(["date.x"]).to_dict(orient='records')):
            states.append( {"i":i **{ key:row[key] for key in ["action_name", "session_name", "task_name"] } } )
            actions.append( row["latency"])

        for i,a2,a1 in zip(count(), [0,0] + actions, [0] + actions):
            states[i]["a-1"] = a1
            states[i]["a-2"] = a2

        episodes.append(Episode(states,actions[:-1]))

    return episodes

df = mindtrails.read_action_log()[["participant_id", "action_name", "date.x", "latency", "session_name", "study_name", "task_name"]]

for participant_id in df["participant_id"].unique():
    
    expert_episodes   = make_episodes(df[df["participant_id"] == participant_id])
    environment_model = mindtrails.Model()
    Q_learner         = make_pipeline(PostStateFeatures(environment_model), DecisionTreeRegressor())
    R_kernel          = mindtrails.Kernel(environment_model)

    policy_learner = DirectEstimateIteration(20, 100, 10, 8, Q_learner, mc_visit="first", duplicate_samples="average")
    reward_learner = KernelProjection(policy_learner, R_kernel, 1, 0.01, 50)

    reward = reward_learner.learn_reward(environment_model, expert_episodes)
