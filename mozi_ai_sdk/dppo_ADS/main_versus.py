from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import app
from absl import flags
from absl import logging

current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.dirname(current_path)
mozi_ai_sdk_path = os.path.dirname(parent_path)

# 打印调试信息
print("Current path:", current_path)
print("Parent path:", parent_path)
print("Mozi_ai_sdk path:", mozi_ai_sdk_path)
print("Original sys.path:", sys.path)

# 添加mozi_ai_sdk的父目录到sys.path
sys.path.insert(0, mozi_ai_sdk_path)
print("Updated sys.path:", sys.path)

# 现在尝试导入
from mozi_ai_sdk.dppo_ADS.envs.env import Environment
# from mozi_ai_sdk.test.dppo.envs import env_remote as environment
from mozi_ai_sdk.dppo_ADS.envs import etc
from mozi_ai_sdk.dppo_ADS.envs.observations import Features
from mozi_ai_sdk.dppo_ADS.envs.tasks import Task
from mozi_ai_sdk.dppo_ADS.utils.utils import print_arguments

# 获取IP和Port
arg = sys.argv
print(arg)
os.environ['MOZIPATH'] = etc.MOZIPATH

FLAGS = flags.FLAGS
flags.DEFINE_string("platform_mode", 'versus', "模式") # 'versus'
flags.DEFINE_integer("num_episodes", 10, "Number of episodes to evaluate.")
flags.DEFINE_enum("agent", 'ppo', ['ppo', 'dqn', 'random', 'keyboard'],
                  "Agent name.")
flags.DEFINE_string("Side", "红方", "side info.")
flags.DEFINE_string("IP", "127.0.0.1", "server IP address.")
flags.DEFINE_string("Port", "6060", "port.")
flags.DEFINE_enum("policy", 'mlp', ['mlp', 'lstm'], "Job type.")
# flags.DEFINE_string("model_path", current_path + "\\checkpoints\\checkpoint", "Filepath to load initial model.")
flags.DEFINE_string("model_path", current_path + "\\bin\\checkpoints\\checkpoint-5000",
                    "Filepath to load initial model.")
flags.FLAGS(sys.argv)


def create_env():
    # if len(arg) != 1:
    #     # pdb.set_trace()
    #     env = environment.Environment(FLAGS.IP, FLAGS.Port, etc.DURATION_INTERVAL)
    # else:
    #     env = Environment(etc.SERVER_IP, etc.SERVER_PORT, etc.PLATFORM, etc.SCENARIO_NAME,
    #                       etc.SIMULATE_COMPRESSION,
    #                       etc.DURATION_INTERVAL,
    #                       etc.SYNCHRONOUS)

    if FLAGS.platform_mode == 'versus':
        print('比赛模式')
        ip = FLAGS.IP
        port = FLAGS.Port
        env = Environment(ip, port, etc.PLATFORM, etc.SCENARIO_NAME, etc.SIMULATE_COMPRESSION, etc.DURATION_INTERVAL,
                          etc.SYNCHRONOUS)

    else:
        print('开发模式')
        env = Environment(etc.SERVER_IP, etc.SERVER_PORT)

    env.start(etc.SERVER_IP, etc.SERVER_PORT)
    scenario = env.reset(FLAGS.Side)
    env = Task(env, scenario, FLAGS.Side)
    env = Features(env, scenario, FLAGS.Side)
    return env


def create_ppo_agent(env):
    import tensorflow as tf
    import multiprocessing
    from mozi_ai_sdk.dppo_ADS.agents.ppo_policies import LstmPolicy, MlpPolicy
    from mozi_ai_sdk.dppo_ADS.agents.ppo_agent import PPOAgent

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()

    policy = {'lstm': LstmPolicy, 'mlp': MlpPolicy}[FLAGS.policy]
    agent = PPOAgent(env=env, policy=policy, model_path=FLAGS.model_path)
    return agent


def evaluate():
    env = create_env()

    if FLAGS.agent == 'ppo':
        agent = create_ppo_agent(env)
    else:
        raise NotImplementedError

    try:
        cum_return = 0.0
        action_counts = [0] * env.action_space.n
        for i in range(FLAGS.num_episodes):
            # pdb.set_trace()
            observation = env.reset()
            agent.reset()
            done, step_id = False, 0
            while not done:
                action = agent.act(observation)
                # print("Step ID: %d	Take Action: %d" % (step_id, action))
                observation, reward, done, _ = env.step(action)
                action_counts[action] += 1
                cum_return += reward
                step_id += 1
            # print_action_distribution(env, action_counts)
            # print("Evaluated %d/%d Episodes Avg Return %f Avg Winning Rate %f" % (
            #     i + 1, FLAGS.num_episodes, cum_return / (i + 1),
            #     ((cum_return / (i + 1)) + 1) / 2.0))
    except KeyboardInterrupt:
        pass
    # finally:
    #     env.close()


def main(argv):
    logging.set_verbosity(logging.ERROR)
    print_arguments(FLAGS)
    evaluate()


if __name__ == '__main__':
    app.run(main)
