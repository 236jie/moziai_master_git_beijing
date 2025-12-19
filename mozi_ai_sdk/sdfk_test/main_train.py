# # 时间 : 2021/2/16 15:58
# # 作者 : Dixit
# # 文件 : main_train.py
# # 说明 :
# # 项目 : 墨子联合作战智能体研发平台
# # 版权 : 北京华戍防务技术有限公司
#
# from ray.tune.schedulers import AsyncHyperBandScheduler
# from ray.tune.suggest import ConcurrencyLimiter
# from ray.tune.suggest.hyperopt import HyperOptSearch
# from ray.rllib.utils.framework import try_import_tf, try_import_torch
#
# import argparse
# from gym.spaces import Discrete, Box, Dict
# import zmq
# import sys
# import torch
# import ray
# from ray import tune
# try:
#     from ray.tune import Callback
# except ImportError:
#     # 如果Ray版本较旧，可能没有Callback类
#     class Callback:
#         pass
# import os
# import time
#
# from ray.remote_handle_docker import stop_docker
# from mozi_ai_sdk.sdfk_test.envs.env_sdfk import SDFKEnv
# import json
#
# file_dir = '/root/logs/'
#
# tf1, tf, tfv = try_import_tf()
# torch, nn = try_import_torch()
#
# parser = argparse.ArgumentParser()
#
# # 集群口令
# parser.add_argument("--address", type=str, default='172.17.94.8:6379')
# parser.add_argument("--redis_password", type=str, default='5241590000000000')
#
# # 训练相关参数
# parser.add_argument("--training_id", type=str, default='test_multi_trials')
# parser.add_argument("--num_gpus", type=int, default=0)
# parser.add_argument("--num_gpus_per_worker", type=int, default=0)
# parser.add_argument("--training_iteration", type=int, default=50000)
# parser.add_argument("--num_samples", type=int, default=1)
# parser.add_argument("--checkpoint_freq", type=int, default=1)
# parser.add_argument("--keep_checkpoints_num", type=int, default=10)
# parser.add_argument("--num_workers", type=int, default=0)  # 0 windows下单机训练
# parser.add_argument("--restore", type=str, default=None)
#
# # 智能体相关参数
# parser.add_argument("--agent_id", type=str, default='robot')
# parser.add_argument("--framework", type=str, default="torch")
# parser.add_argument("--vf_share_layers", type=bool, default=True)
# parser.add_argument("--vf_loss_coeff", type=float, default=1.0)
# parser.add_argument("--kl_coeff", type=float, default=0.2)
# parser.add_argument("--clip_param", type=float, default=0.3)
# parser.add_argument("--vf_clip_param", type=float, default=10)
# parser.add_argument("--lr_min", type=float, default=5e-6)
# parser.add_argument("--lr_max", type=float, default=5e-5)
# parser.add_argument("--num_sgd_iter", type=int, default=100)
# parser.add_argument("--sgd_minibatch_size", type=int, default=128)
# parser.add_argument("--rollout_fragment_length", type=int, default=512)
# parser.add_argument("--train_batch_size", type=int, default=-1)
# parser.add_argument("--side", type=str, default="红方")
#
# parser.add_argument("--Lambda", type=float, default=0.98)
# parser.add_argument("--algorithm", type=str, default="DDPPO")
#
# # 需确认
# parser.add_argument("--torch", action="store_true")
# parser.add_argument("--as-test", action="store_true")
# parser.add_argument("--stop-iters", type=int, default=50000)
# parser.add_argument("--stop-timesteps", type=int, default=1000000)
# parser.add_argument("--stop-reward", type=float, default=1.5)
# parser.add_argument("--platform_mode", type=str, default='eval')  # 'eval' windows下单机训练
# parser.add_argument("--mozi_server_path", type=str, default=r'D:\huashuanzhuang\mozilianhe\Mozi\MoziServer\bin')
#
# # zmq init
# zmq_context = zmq.Context()
#
# # 创建的docker个数应该是num_workers+1，比如num_workers=3，那么需要创建4个docker
# SERVER_DOCKER_DICT = {'127.0.0.1': 1, }  # {'8.140.121.210': 2, '123.57.137.210': 2}
#
#
# def reset_training_docker(_training_id):
#     """
#     功能：重启训练docker
#     作者：张志高
#     时间：2021-2-16
#     """
#     try:
#         message = {}
#         message['zmq_command'] = 'reset_training_docker'
#         message['training_id'] = _training_id
#         socket_to = g_zmq_manager.send_message_to_backend(message)
#         recv_msg = socket_to.recv_pyobj()
#         assert type(recv_msg) == str
#         if 'OK' in recv_msg:
#             print(f'重启训练docker成功，训练ID: {_training_id}')
#         else:
#             sys.exit(1)
#     except Exception:
#         print(f'重启训练docker失败，训练ID: {_training_id}')
#         sys.exit(1)
#
#
# def start_tune(training_id=None,
#                num_gpus=None,
#                num_gpus_per_worker=None,
#                num_workers=10,
#                training_iteration=None,
#                num_samples=1,
#                checkpoint_freq=None,
#                keep_checkpoints_num=None,
#                framework=None,
#                model=None,
#                vf_share_layers=None,
#                vf_loss_coeff=1.0,
#                kl_coeff=0.2,
#                vf_clip_param=10.0,
#                Lambda=None,
#                clip_param=0.3,
#                lr_min=5e-6,
#                lr_max=5e-4,
#                num_sgd_iter=100,
#                sgd_minibatch_size=256,
#                rollout_fragment_length=512,
#                train_batch_size=-1,
#                side_name=None,
#                restore=None,
#                platform_mode=None,
#                # 内部参数
#                algorithm_name='DDPPO',
#                action_size=None,
#                obs_size=None,
#                log_to_file=file_dir,
#                agent_id=None):
#     act_space = Discrete(action_size)
#     obs_space = Dict({"obs": Box(float("-inf"), float("inf"), shape=(obs_size,)),
#                       # "action_mask": Box(0, 1, shape=(action_size,)),
#                       })
#
#     config = {"env": SDFKEnv,
#               "env_config": {'mode': platform_mode,  # 'train'/'development'/'eval'
#                              # 'sever_docker_dict': SERVER_DOCKER_DICT,  # {'8.140.121.210': 2, '123.57.81.172': 2}
#                              'avail_docker_ip_port': ['127.0.0.1:6060', ],  # windows下单机训练
#                              'side_name': side_name,
#                              'enemy_side_name': '蓝方',
#                              'action_dim': action_size,
#                              'obs_dim': obs_size,
#                              'training_id': training_id,
#                              },
#               # "monitor": True,
#               # "ignore_worker_failures": True,
#               # "log_level": "DEBUG",
#               "num_gpus": num_gpus,
#               "num_gpus_per_worker": num_gpus_per_worker,
#               # "queue_trials": True,
#               "framework": framework,
#               "model": {"use_lstm": False,
#                         # "custom_model": "mask_model",
#                         "max_seq_len": 64,
#                         # Size of the LSTM cell.
#                         "lstm_cell_size": 256,
#                         # Whether to feed a_{t-1}, r_{t-1} to LSTM.
#                         "lstm_use_prev_action_reward": True,
#                         },
#               'multiagent': {
#                   'agent_0': (obs_space, act_space, {"gamma": 0.99}),
#               },
#               "lambda": 0.98,
#               "vf_share_layers": True,
#               "vf_loss_coeff": vf_loss_coeff,
#               'entropy_coeff': 0.0,
#               "kl_coeff": kl_coeff,
#               "vf_clip_param": vf_clip_param,
#               "clip_param": clip_param,
#               "lr": tune.uniform(lr_min, lr_max),
#               "num_sgd_iter": num_sgd_iter,
#               "sgd_minibatch_size": sgd_minibatch_size,
#               "rollout_fragment_length": rollout_fragment_length,
#               "num_envs_per_worker": 1,
#               "train_batch_size": train_batch_size,
#               "batch_mode": "truncate_episodes",
#               "num_workers": num_workers,
#               }
#     if platform_mode == 'train':
#         config['env_config']['schedule_addr'] = BACKEND_SERVER_IP
#         config['env_config']['schedule_port'] = BACKEND_SERVER_PORT
#
#     stop = {
#         "training_iteration": training_iteration,
#     }
#     best_trial = None
#     best_config = None
#     try:
#         # Windows 上单机训练且 num_samples=1 时，不需要搜索器，避免 searcher-state 重命名冲突
#         algo = None
#         scheduler = AsyncHyperBandScheduler(max_t=1000)
#         if num_samples and num_samples > 1:
#             algo = HyperOptSearch()
#             algo = ConcurrencyLimiter(algo, max_concurrent=1)
#         if platform_mode == 'train':
#             result_dir = os.path.join(TRAINING_RESULT_PATH, agent_id, 'result')
#         elif platform_mode == 'development':
#             result_dir = None
#         elif platform_mode == 'eval':
#             result_dir = None
#         else:
#             raise NotImplementedError
#         run_name = training_id
#         # 在 Windows 下避免旧目录文件残留导致重命名失败，追加时间戳形成唯一目录
#         if sys.platform.startswith('win'):
#             run_name = f"{training_id}_{int(time.time())}"
#
#         # ===== 创建训练回调：输出训练轮数和分隔线 =====
#         class TrainingProgressCallback(Callback):
#             def on_trial_result(self, iteration, trials, trial, result, **info):
#                 """每次训练迭代后的回调"""
#                 episode_reward_mean = result.get('episode_reward_mean', 0)
#                 episode_reward_max = result.get('episode_reward_max', 0)
#                 episode_reward_min = result.get('episode_reward_min', 0)
#
#                 # 输出训练轮数（用分隔线）
#                 print("=" * 100)
#                 print(f"训练轮数: {iteration} | 平均奖励: {episode_reward_mean:.2f} | "
#                       f"最高奖励: {episode_reward_max:.2f} | 最低奖励: {episode_reward_min:.2f}")
#                 print("=" * 100)
#                 return False  # 不停止训练
#
#         callback = TrainingProgressCallback()
#
#         results = tune.run(algorithm_name,
#                            name=run_name,
#                            metric="episode_reward_mean",
#                            mode="max",
#                            local_dir=result_dir,
#                            search_alg=algo,
#                            scheduler=scheduler,
#                            num_samples=num_samples,
#                            checkpoint_freq=checkpoint_freq,
#                            keep_checkpoints_num=keep_checkpoints_num,
#                            config=config,
#                            restore=restore,
#                            callbacks=[callback],
#                            # log_to_file=True,
#                            # max_failures=3,
#                            # resume=True,
#                            # queue_trials=False,
#                            # stop=stop
#                            )
#
#         # ===== 训练后处理：记录最好的结果 =====
#         try:
#             # 从结果中提取所有iteration的信息
#             best_results_list = []
#             if hasattr(results, 'trials') and len(results.trials) > 0:
#                 for trial in results.trials:
#                     if hasattr(trial, 'results') and trial.results:
#                         for result in trial.results:
#                             iteration = result.get('training_iteration', 0)
#                             reward = result.get('episode_reward_mean', 0)
#                             info = {
#                                 'protected_facilities': result.get('custom_metrics', {}).get('protected_facilities_mean', 0),
#                                 'intercepted_missiles': result.get('custom_metrics', {}).get('intercepted_missiles_mean', 0),
#                             }
#                             best_results_list.append((reward, iteration, info))
#
#             # 排序并取最好的10个
#             if best_results_list:
#                 best_results_list.sort(key=lambda x: x[0], reverse=True)
#                 top_10 = best_results_list[:10]
#
#                 # 保存到文件
#                 data = [
#                     {
#                         'rank': i + 1,
#                         'reward': reward,
#                         'iteration': iteration,
#                         'info': info
#                     }
#                     for i, (reward, iteration, info) in enumerate(top_10)
#                 ]
#                 os.makedirs("./training_logs", exist_ok=True)
#                 with open("./training_logs/best_results.json", 'w', encoding='utf-8') as f:
#                     json.dump(data, f, indent=2, ensure_ascii=False)
#
#                 print(f"\n保存最好的10次训练结果到 training_logs/best_results.json")
#         except Exception as e:
#             print(f"处理训练结果时出错: {e}")
#
#         best_trial = results.get_best_trial('episode_reward_mean')
#         best_config = results.get_best_config('episode_reward_mean')
#         print(best_trial)
#         print(best_config)
#     except Exception as e:
#         print(f'训练时发生异常：{str(e)}')
#         # 后续放开 张志高 2021-2-16
#         # reset_training_docker(training_id)
#         if platform_mode == 'development':
#             stop_docker(SERVER_DOCKER_DICT)
#         # import traceback
#         # traceback.print_exc()
#     return best_trial, best_config
#
#
# if __name__ == '__main__':
#
#     args = parser.parse_args()
#     if args.algorithm.upper() == 'DDPPO' and args.num_workers == 0:
#         print("警告：DDPPO 要求 num_workers > 0。当前 num_workers=0。"
#           " 自动将算法切换为 PPO 以便本地训练。若要使用 DDPPO，请传入 --num_workers >= 1")
#         args.algorithm = 'PPO'
#
#     if args.train_batch_size is None or args.train_batch_size <= 0:
#         args.train_batch_size = max(args.rollout_fragment_length, args.sgd_minibatch_size)
#     if args.platform_mode == 'train':
#         # from ray.managers.config import *
#         # from ray.managers.utils import *
#         # from ray.managers.zmq_manager import g_zmq_manager
#         # g_zmq_manager.register_me(args.training_id)
#         # g_zmq_manager.start_listen_thread()
#         ray.init(address=args.address, _redis_password=args.redis_password)
#     elif args.platform_mode == 'development':
#         ray.init(address="auto")
#         # ray.init(local_mode=True)
#     elif args.platform_mode == 'eval':  # windows下单机训练
#         os.environ['MOZIPATH'] = args.mozi_server_path
#         # ray.init(local_mode=True)
#         ray.init(num_gpus=0)
#     else:
#         raise NotImplementedError
#
#     start_tune(training_id=args.training_id,
#                num_gpus=args.num_gpus,
#                num_gpus_per_worker=args.num_gpus_per_worker,
#                num_workers=args.num_workers,
#                training_iteration=args.training_iteration,
#                num_samples=args.num_samples,  # 警告, 该值为并行实验个数，当前只能传1，
#                checkpoint_freq=args.checkpoint_freq,
#                keep_checkpoints_num=args.keep_checkpoints_num,
#                framework=args.framework,
#                vf_share_layers=args.vf_share_layers,
#                vf_loss_coeff=args.vf_loss_coeff,
#                kl_coeff=args.kl_coeff,
#                vf_clip_param=args.vf_clip_param,
#                Lambda=args.Lambda,
#                clip_param=args.clip_param,
#                lr_min=args.lr_min,
#                lr_max=args.lr_max,
#                num_sgd_iter=args.num_sgd_iter,
#                sgd_minibatch_size=args.sgd_minibatch_size,
#                rollout_fragment_length=args.rollout_fragment_length,
#                train_batch_size=args.train_batch_size,
#                side_name=args.side,
#                restore=args.restore,
#                platform_mode=args.platform_mode,
#                # 内部参数
#                algorithm_name=args.algorithm,
#                action_size=73,
#                obs_size=20,
#                log_to_file=file_dir,
#                agent_id=args.agent_id)
#
#     print('训练结束')
#
#
#

# 时间 : 2021/2/16 15:58
# 作者 : Dixit
# 文件 : main_train.py
# 说明 :
# 项目 : 墨子联合作战智能体研发平台
# 版权 : 北京华戍防务技术有限公司

from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.rllib.utils.framework import try_import_tf, try_import_torch

import argparse
from gym.spaces import Discrete, Box, Dict
import zmq
import sys
import torch
import ray
from ray import tune
import os
import time

from ray.remote_handle_docker import stop_docker
from mozi_ai_sdk.sdfk_test.envs.env_sdfk import SDFKEnv
import json

file_dir = '/root/logs/'

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()

# 集群口令
parser.add_argument("--address", type=str, default='172.17.94.8:6379')
parser.add_argument("--redis_password", type=str, default='5241590000000000')

# 训练相关参数
parser.add_argument("--training_id", type=str, default='test_multi_trials')
parser.add_argument("--num_gpus", type=int, default=0)
parser.add_argument("--num_gpus_per_worker", type=int, default=0)
parser.add_argument("--training_iteration", type=int, default=50000)
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--checkpoint_freq", type=int, default=1)
parser.add_argument("--keep_checkpoints_num", type=int, default=10)
parser.add_argument("--num_workers", type=int, default=0)  # 0 windows下单机训练
parser.add_argument("--restore", type=str, default=None)

# 智能体相关参数
parser.add_argument("--agent_id", type=str, default='robot')
parser.add_argument("--framework", type=str, default="torch")
parser.add_argument("--vf_share_layers", type=bool, default=True)
parser.add_argument("--vf_loss_coeff", type=float, default=1.0)
parser.add_argument("--kl_coeff", type=float, default=0.2)
parser.add_argument("--clip_param", type=float, default=0.3)
parser.add_argument("--vf_clip_param", type=float, default=10)
parser.add_argument("--lr_min", type=float, default=5e-6)
parser.add_argument("--lr_max", type=float, default=5e-5)
parser.add_argument("--num_sgd_iter", type=int, default=100)
parser.add_argument("--sgd_minibatch_size", type=int, default=128)
parser.add_argument("--rollout_fragment_length", type=int, default=512)
parser.add_argument("--train_batch_size", type=int, default=-1)
parser.add_argument("--side", type=str, default="红方")

parser.add_argument("--Lambda", type=float, default=0.98)
parser.add_argument("--algorithm", type=str, default="DDPPO")

# 需确认
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=50000)
parser.add_argument("--stop-timesteps", type=int, default=1000000)
parser.add_argument("--stop-reward", type=float, default=1.5)
parser.add_argument("--platform_mode", type=str, default='eval')  # 'eval' windows下单机训练
parser.add_argument("--mozi_server_path", type=str, default=r'D:\huashuanzhuang\mozilianhe\Mozi\MoziServer\bin')

# zmq init
zmq_context = zmq.Context()

# 创建的docker个数应该是num_workers+1，比如num_workers=3，那么需要创建4个docker
SERVER_DOCKER_DICT = {'127.0.0.1': 1, }  # {'8.140.121.210': 2, '123.57.137.210': 2}


def reset_training_docker(_training_id):
    """
    功能：重启训练docker
    作者：张志高
    时间：2021-2-16
    """
    try:
        message = {}
        message['zmq_command'] = 'reset_training_docker'
        message['training_id'] = _training_id
        socket_to = g_zmq_manager.send_message_to_backend(message)
        recv_msg = socket_to.recv_pyobj()
        assert type(recv_msg) == str
        if 'OK' in recv_msg:
            print(f'重启训练docker成功，训练ID: {_training_id}')
        else:
            sys.exit(1)
    except Exception:
        print(f'重启训练docker失败，训练ID: {_training_id}')
        sys.exit(1)


def start_tune(training_id=None,
               num_gpus=None,
               num_gpus_per_worker=None,
               num_workers=10,
               training_iteration=None,
               num_samples=1,
               checkpoint_freq=None,
               keep_checkpoints_num=None,
               framework=None,
               model=None,
               vf_share_layers=None,
               vf_loss_coeff=1.0,
               kl_coeff=0.2,
               vf_clip_param=10.0,
               Lambda=None,
               clip_param=0.3,
               lr_min=5e-6,
               lr_max=5e-4,
               num_sgd_iter=100,
               sgd_minibatch_size=256,
               rollout_fragment_length=512,
               train_batch_size=-1,
               side_name=None,
               restore=None,
               platform_mode=None,
               # 内部参数
               algorithm_name='DDPPO',
               action_size=None,
               obs_size=None,
               log_to_file=file_dir,
               agent_id=None):
    act_space = Discrete(action_size)
    obs_space = Dict({"obs": Box(float("-inf"), float("inf"), shape=(obs_size,)),
                      # "action_mask": Box(0, 1, shape=(action_size,)),
                      })

    config = {"env": SDFKEnv,
              "env_config": {'mode': platform_mode,  # 'train'/'development'/'eval'
                             # 'sever_docker_dict': SERVER_DOCKER_DICT,  # {'8.140.121.210': 2, '123.57.81.172': 2}
                             'avail_docker_ip_port': ['127.0.0.1:6060', ],  # windows下单机训练
                             'side_name': side_name,
                             'enemy_side_name': '蓝方',
                             'action_dim': action_size,
                             'obs_dim': obs_size,
                             'training_id': training_id,
                             },
              # "monitor": True,
              # "ignore_worker_failures": True,
              # "log_level": "DEBUG",
              "num_gpus": num_gpus,
              "num_gpus_per_worker": num_gpus_per_worker,
              # "queue_trials": True,
              "framework": framework,
              "model": {"use_lstm": False,
                        # "custom_model": "mask_model",
                        "max_seq_len": 64,
                        # Size of the LSTM cell.
                        "lstm_cell_size": 256,
                        # Whether to feed a_{t-1}, r_{t-1} to LSTM.
                        "lstm_use_prev_action_reward": True,
                        },
              'multiagent': {
                  'agent_0': (obs_space, act_space, {"gamma": 0.99}),
              },
              "lambda": 0.98,
              "vf_share_layers": True,
              "vf_loss_coeff": vf_loss_coeff,
              'entropy_coeff': 0.0,
              "kl_coeff": kl_coeff,
              "vf_clip_param": vf_clip_param,
              "clip_param": clip_param,
              "lr": tune.uniform(lr_min, lr_max),
              "num_sgd_iter": num_sgd_iter,
              "sgd_minibatch_size": sgd_minibatch_size,
              "rollout_fragment_length": rollout_fragment_length,
              "num_envs_per_worker": 1,
              "train_batch_size": train_batch_size,
              "batch_mode": "truncate_episodes",
              "num_workers": num_workers,
              }
    if platform_mode == 'train':
        config['env_config']['schedule_addr'] = BACKEND_SERVER_IP
        config['env_config']['schedule_port'] = BACKEND_SERVER_PORT

    stop = {
        "training_iteration": training_iteration,
    }
    best_trial = None
    best_config = None
    try:
        # Windows 上单机训练且 num_samples=1 时，不需要搜索器，避免 searcher-state 重命名冲突
        algo = None
        scheduler = AsyncHyperBandScheduler(max_t=1000)
        if num_samples and num_samples > 1:
            algo = HyperOptSearch()
            algo = ConcurrencyLimiter(algo, max_concurrent=1)
        if platform_mode == 'train':
            result_dir = os.path.join(TRAINING_RESULT_PATH, agent_id, 'result')
        elif platform_mode == 'development':
            result_dir = None
        elif platform_mode == 'eval':
            result_dir = None
        else:
            raise NotImplementedError
        run_name = training_id
        # 在 Windows 下避免旧目录文件残留导致重命名失败，追加时间戳形成唯一目录
        if sys.platform.startswith('win'):
            run_name = f"{training_id}_{int(time.time())}"

        # ===== 移除低版本不支持的 Callback 类 =====
        # 替代方案：训练过程中打印关键指标（通过 tune.run 的 verbose 参数）

        results = tune.run(algorithm_name,
                           name=run_name,
                           metric="episode_reward_mean",
                           mode="max",
                           local_dir=result_dir,
                           search_alg=algo,
                           scheduler=scheduler,
                           num_samples=num_samples,
                           checkpoint_freq=checkpoint_freq,
                           keep_checkpoints_num=keep_checkpoints_num,
                           config=config,
                           restore=restore,
                           verbose=2,  # 增加日志输出级别，替代Callback打印进度
                           stop=stop  # 补全stop参数（原代码注释掉了，建议加上）
                           )

        # ===== 训练后处理：记录最好的结果 =====
        try:
            # 从结果中提取所有iteration的信息
            best_results_list = []
            if hasattr(results, 'trials') and len(results.trials) > 0:
                for trial in results.trials:
                    if hasattr(trial, 'results') and trial.results:
                        for result in trial.results:
                            iteration = result.get('training_iteration', 0)
                            reward = result.get('episode_reward_mean', 0)
                            info = {
                                'protected_facilities': result.get('custom_metrics', {}).get(
                                    'protected_facilities_mean', 0),
                                'intercepted_missiles': result.get('custom_metrics', {}).get(
                                    'intercepted_missiles_mean', 0),
                            }
                            best_results_list.append((reward, iteration, info))

            # 排序并取最好的10个
            if best_results_list:
                best_results_list.sort(key=lambda x: x[0], reverse=True)
                top_10 = best_results_list[:10]

                # 保存到文件
                data = [
                    {
                        'rank': i + 1,
                        'reward': reward,
                        'iteration': iteration,
                        'info': info
                    }
                    for i, (reward, iteration, info) in enumerate(top_10)
                ]
                os.makedirs("./training_logs", exist_ok=True)
                with open("./training_logs/best_results.json", 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                print(f"\n保存最好的10次训练结果到 training_logs/best_results.json")
        except Exception as e:
            print(f"处理训练结果时出错: {e}")

        best_trial = results.get_best_trial('episode_reward_mean')
        best_config = results.get_best_config('episode_reward_mean')
        print(best_trial)
        print(best_config)
    except Exception as e:
        print(f'训练时发生异常：{str(e)}')
        # 后续放开 张志高 2021-2-16
        # reset_training_docker(training_id)
        if platform_mode == 'development':
            stop_docker(SERVER_DOCKER_DICT)
        # import traceback
        # traceback.print_exc()
    return best_trial, best_config


if __name__ == '__main__':

    args = parser.parse_args()
    if args.algorithm.upper() == 'DDPPO' and args.num_workers == 0:
        print("警告：DDPPO 要求 num_workers > 0。当前 num_workers=0。"
              " 自动将算法切换为 PPO 以便本地训练。若要使用 DDPPO，请传入 --num_workers >= 1")
        args.algorithm = 'PPO'

    if args.train_batch_size is None or args.train_batch_size <= 0:
        args.train_batch_size = max(args.rollout_fragment_length, args.sgd_minibatch_size)
    if args.platform_mode == 'train':
        # from ray.managers.config import *
        # from ray.managers.utils import *
        # from ray.managers.zmq_manager import g_zmq_manager
        # g_zmq_manager.register_me(args.training_id)
        # g_zmq_manager.start_listen_thread()
        ray.init(address=args.address, _redis_password=args.redis_password)
    elif args.platform_mode == 'development':
        ray.init(address="auto")
        # ray.init(local_mode=True)
    elif args.platform_mode == 'eval':  # windows下单机训练
        os.environ['MOZIPATH'] = args.mozi_server_path
        # ray.init(local_mode=True)
        ray.init(num_gpus=0)
    else:
        raise NotImplementedError

    start_tune(training_id=args.training_id,
               num_gpus=args.num_gpus,
               num_gpus_per_worker=args.num_gpus_per_worker,
               num_workers=args.num_workers,
               training_iteration=args.training_iteration,
               num_samples=args.num_samples,  # 警告, 该值为并行实验个数，当前只能传1，
               checkpoint_freq=args.checkpoint_freq,
               keep_checkpoints_num=args.keep_checkpoints_num,
               framework=args.framework,
               vf_share_layers=args.vf_share_layers,
               vf_loss_coeff=args.vf_loss_coeff,
               kl_coeff=args.kl_coeff,
               vf_clip_param=args.vf_clip_param,
               Lambda=args.Lambda,
               clip_param=args.clip_param,
               lr_min=args.lr_min,
               lr_max=args.lr_max,
               num_sgd_iter=args.num_sgd_iter,
               sgd_minibatch_size=args.sgd_minibatch_size,
               rollout_fragment_length=args.rollout_fragment_length,
               train_batch_size=args.train_batch_size,
               side_name=args.side,
               restore=args.restore,
               platform_mode=args.platform_mode,
               # 内部参数
               algorithm_name=args.algorithm,
               action_size=73,
               obs_size=20,
               log_to_file=file_dir,
               agent_id=args.agent_id)

    print('训练结束')
