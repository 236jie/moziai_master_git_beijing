# 时间 ： 2020/12/19 17:28
# 作者 ： Dixit
# 文件 ： tune_evaluate.py
# 项目 ： moziAI_nlz
# 版权 ： 北京华戍防务技术有限公司

import configparser
import os
import shutil
import sys

import ray
from gym.spaces import Discrete, Box, Dict
from ray.rllib.agents.ppo import PPOTrainer

from mozi_ai_sdk.nlz_wrj.envs.wrj_env_eval import WRJ


def resource_path(fileName):
    # 获取exe解压后的文件
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, fileName)
    return os.path.join(fileName)

# 获取IP和Port
path_exe = sys.argv[0]  # 生成的exe文件所在的地址
src = os.path.join(os.path.dirname(path_exe), 'peizhi.ini')
src2 = os.path.join(os.path.dirname(path_exe), 'redis-server.exe')
src3 = os.path.join(os.path.dirname(path_exe), 'libray_redis_module.so')

# 把exe的配置文件复制到exe解压后的临时文件夹中
if hasattr(sys, "_MEIPASS"):
    des = os.path.join(sys._MEIPASS, 'peizhi.ini')
    # des2 = os.path.join(sys._MEIPASS,'ray') # 没有这个文件夹_raylet.pyd
    shutil.copyfile(src, des)
    # shutil.copy(src2, des2)
    # shutil.copy(src3, des2)

# 解析配置文件
config = configparser.ConfigParser()  # 类实例化
config_path = resource_path('peizhi.ini')
print(f'tmp临时文件中配置文件的地址{config_path}')


# config.read(config_path)
config.read(config_path, encoding="utf-8")
mozi_path = config['select']['MOZIPATH']
print(mozi_path)
# mozi_path = 'D:\\mozi_server\\Mozi\\MoziServer\\bin'
os.environ['MOZIPATH'] = mozi_path

# 获取训练好的模型
checkpoint = resource_path('checkpoint-99')
print(f'资源文件临时路径为{checkpoint}')


def get_ppo_agent(env, checkpoint, config):
    trainer = PPOTrainer(env=env, config=config)
    trainer.restore(checkpoint)
    return trainer


class Evaluate(object):
    def __init__(self, env, checkpoint, config, evaluate_episodes, explore=True):
        self.agent = get_ppo_agent(env, checkpoint, config)
        self.evaluate_episodes = evaluate_episodes
        self.env = env(config['env_config'])
        self.explore = explore

    def eval(self):
        evaluate_rewards = []

        for episode in range(self.evaluate_episodes):
            obs = self.env.reset()
            rewards = 0
            while True:
                if type(obs) is dict:
                    action = {}
                    for agent_name in obs:
                        action[agent_name] = self.agent.compute_action(obs[agent_name], explore=self.explore)
                else:
                    action = self.agent.compute_action(obs, explore=self.explore)
                    action = action[0][0]
                obs_next, reward, done, _ = self.env.step(action)
                obs = obs_next
                if type(reward) is dict:
                    for k, v in reward.items():
                        rewards += v
                else:
                    reward += reward
                if ((type(done) is dict) and done['__all__']) or ((type(done) is bool) and done):
                    evaluate_rewards.append(rewards)
                    break

        average_reward = sum(evaluate_rewards) / len(evaluate_rewards)
        return average_reward


ray.init(local_mode=True)

env = WRJ
evaluate_episodes = 10
n_dims = 2 * 8 + 3 * 8 + 3 + 39
act_space = Discrete(6 * 5 + 3 * 3)
obs_space = Dict({"obs": Box(float('-inf'), float('inf'), shape=(n_dims,))})
config = {  # "env": WRJ,
    "env_config": {'mode': 'eval', 'avail_docker_ip_port': [6060]},
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
    # "lr": tune.uniform(1e-4, 1e-2),
    # These params start off randomly drawn from a set.
    # "num_sgd_iter": 10,
    # "sgd_minibatch_size": tune.choice([1024, 2048]),
    # "train_batch_size": sample_from(
    #     lambda spec: random.choice([10000, 20000, 40000])),
    # "train_batch_size": 16384,
    "batch_mode": "complete_episodes",  # 'truncate_episodes'
    "num_workers": 0,
    # "evaluation_interval": 1,
    # "evaluation_num_episodes": 1,
    # "evaluation_num_workers": 0
}
evaluate = Evaluate(env, checkpoint, config, evaluate_episodes, explore=False)


evaluate.eval()
