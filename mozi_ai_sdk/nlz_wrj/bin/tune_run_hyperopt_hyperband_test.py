import time

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
import hyperopt

import argparse
import gym
from gym.spaces import Discrete, Box, Dict
import numpy as np
import random
import time
import os

import ray
from ray import tune
from ray.tune import grid_search
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune import run, sample_from
from ray.tune.schedulers import PopulationBasedTraining
from mozi_ai_sdk.test.nlz_wrj.envs.wrj_env import WRJ

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=1)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=0.7)

ray.init(local_mode=True)
# ray.init()

n_dims = 2*8 + 3*8 + 3 + 39
act_space = Discrete(6*5 + 3*3)
obs_space = Dict({"obs": Box(float('-inf'), float('inf'), shape=(n_dims,))})

# TODO docker manager

if __name__ == "__main__":
    args = parser.parse_args()
    # ray.init(address='auto', _redis_password='5241590000000000')
    # ray.init()
    # ray.init()

    config = {"env": WRJ,
              "env_config": {'mode': 'train'},
              # "num_gpus": 1,
              "framework": 'torch',
              'multiagent': {
                  'agent_0': (obs_space, act_space, {"gamma": 0.99}),
                  # 'fighter_1': (obs_space, act_space, {"gamma": 0.99}),
                  # 'fighter_2': (obs_space, act_space, {"gamma": 0.99}),
              },
              "vf_share_layers": True,
              "vf_loss_coeff": 1e-5,  # 1e-2, 5e-4,
              # "lr": grid_search([1e-2]),#, 1e-4, 1e-6]),  # try different lrs
              "kl_coeff": 1.0,
              "vf_clip_param": 1e3,
              # These params are tuned from a fixed starting value.
              "lambda": 0.95,
              "clip_param": 0.2,
              "lr": tune.uniform(1e-4, 1e-2),
              # These params start off randomly drawn from a set.
              "num_sgd_iter": 10,
              "sgd_minibatch_size": 1024,
              # "train_batch_size": sample_from(
              #     lambda spec: random.choice([10000, 20000, 40000])),
              "train_batch_size": 4096,
              "batch_mode": "complete_episodes",  # 'truncate_episodes'
              "num_workers": 1,
              # "evaluation_interval": 1,
              # "evaluation_num_episodes": 1,
              # "evaluation_num_workers": 0
              }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    algo = HyperOptSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=2)
    scheduler = AsyncHyperBandScheduler()
    results = tune.run(args.run,
                       name="hyper_wrj_test",
                       metric="episode_reward_mean",
                       mode="max",
                       search_alg=algo,
                       scheduler=scheduler,
                       num_samples=12,
                       config=config,
                       stop=stop
                       )

    # if args.as_test:
    #     check_learning_achieved(results, args.stop_reward)
    # ray.shutdown()
