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
from mozi_ai_sdk.sdfk_test.model_manager import ModelManager
import json
import signal
import threading
import glob
import re

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
# parser.add_argument("--training_iteration", type=int, default=20)# 训练迭代次数（已修改为20次，对应20个episode）
parser.add_argument("--training_iteration", type=int, default=100)# 训练迭代次数（已修改为20次，对应20个episode）

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
parser.add_argument("--stop-iters", type=int, default=100) # 最大训练迭代次数（已修改为20次，对应20个episode）
parser.add_argument("--stop-timesteps", type=int, default=1000000)# 最大时间步数
parser.add_argument("--stop-reward", type=float, default=1.5) # 目标奖励阈值
parser.add_argument("--platform_mode", type=str, default='eval')  # 'eval' windows下单机训练
parser.add_argument("--mozi_server_path", type=str, default=r'D:\huashuanzhuang\mozilianhe\Mozi\MoziServer\bin')

# zmq init
zmq_context = zmq.Context()

# 创建的docker个数应该是num_workers+1，比如num_workers=3，那么需要创建4个docker
SERVER_DOCKER_DICT = {'127.0.0.1': 1, }  # {'8.140.121.210': 2, '123.57.137.210': 2}


def parse_episode_info_from_logs(trial_dir):
    """
    从trial目录的日志文件中解析episode信息
    
    Args:
        trial_dir: trial目录路径
    
    Returns:
        dict: {iteration: {'protected_count': int, 'missile_consumption': dict, 'missile_cost': float}}
    """
    episode_info = {}
    
    # 查找可能的日志文件
    log_files = []
    if trial_dir and os.path.exists(trial_dir):
        # 查找所有可能的日志文件
        for file in os.listdir(trial_dir):
            if file.endswith('.log') or file.endswith('.txt') or 'log' in file.lower():
                log_files.append(os.path.join(trial_dir, file))
    
    # 如果没有找到日志文件，尝试从标准输出中解析（如果Ray保存了）
    # 或者从result.json中查找相关信息
    
    # 正则表达式
    re_protected = re.compile(r"红方地面核心设施还剩下(\d+)个")
    re_missile_consumption = re.compile(r"本局导弹消耗:\s*C-400=(-?\d+),\s*HQ-9A=(-?\d+),\s*HQ-12=(-?\d+),\s*总费用=(-?\d+\.?\d*)")
    
    # 从result.json中查找（如果包含日志信息）
    result_json_path = os.path.join(trial_dir, "result.json") if trial_dir else None
    if result_json_path and os.path.exists(result_json_path):
        try:
            with open(result_json_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                current_iteration = None
                last_protected = None
                last_missile_info = None
                
                for line in lines:
                    if not line.strip():
                        continue
                    
                    # 先尝试解析JSON
                    try:
                        result = json.loads(line)
                        iteration = result.get('training_iteration', 0)
                        if iteration > 0:
                            current_iteration = iteration
                            # 如果之前有解析到的信息，保存到当前iteration
                            if last_protected is not None or last_missile_info is not None:
                                if current_iteration not in episode_info:
                                    episode_info[current_iteration] = {
                                        'protected_count': last_protected if last_protected is not None else 0,
                                        'missile_consumption': last_missile_info['missile_consumption'] if last_missile_info else {'C-400': 0, 'HQ-9A': 0, 'HQ-12': 0},
                                        'missile_cost': last_missile_info['missile_cost'] if last_missile_info else 0.0
                                    }
                                else:
                                    if last_protected is not None:
                                        episode_info[current_iteration]['protected_count'] = last_protected
                                    if last_missile_info:
                                        episode_info[current_iteration]['missile_consumption'] = last_missile_info['missile_consumption']
                                        episode_info[current_iteration]['missile_cost'] = last_missile_info['missile_cost']
                            last_protected = None
                            last_missile_info = None
                        continue
                    except json.JSONDecodeError:
                        pass
                    
                    # 如果不是JSON，尝试从行中解析信息
                    # 查找保护目标数量（查找最后一个出现的，因为可能有多行）
                    m = re_protected.search(line)
                    if m:
                        try:
                            last_protected = int(m.group(1))
                        except ValueError:
                            pass
                    
                    # 查找导弹消耗（查找最后一个出现的）
                    m = re_missile_consumption.search(line)
                    if m:
                        try:
                            c400 = int(m.group(1))
                            hq9a = int(m.group(2))
                            hq12 = int(m.group(3))
                            cost = float(m.group(4))
                            last_missile_info = {
                                'missile_consumption': {'C-400': c400, 'HQ-9A': hq9a, 'HQ-12': hq12},
                                'missile_cost': cost
                            }
                        except (ValueError, IndexError):
                            pass
                
                # 处理最后一个iteration的信息
                if current_iteration and (last_protected is not None or last_missile_info is not None):
                    if current_iteration not in episode_info:
                        episode_info[current_iteration] = {
                            'protected_count': last_protected if last_protected is not None else 0,
                            'missile_consumption': last_missile_info['missile_consumption'] if last_missile_info else {'C-400': 0, 'HQ-9A': 0, 'HQ-12': 0},
                            'missile_cost': last_missile_info['missile_cost'] if last_missile_info else 0.0
                        }
                    else:
                        if last_protected is not None:
                            episode_info[current_iteration]['protected_count'] = last_protected
                        if last_missile_info:
                            episode_info[current_iteration]['missile_consumption'] = last_missile_info['missile_consumption']
                            episode_info[current_iteration]['missile_cost'] = last_missile_info['missile_cost']
        except Exception as e:
            print(f"解析result.json失败: {e}")
            import traceback
            traceback.print_exc()
    
    return episode_info


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
                             # 'action_bias': [0.425, 0.425, 0.15],  # 85%拦截，15%不拦截
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

    # 自定义停止函数：完成20个episode后停止
    def stop_fn(trial_id, result):
        """自定义停止条件：完成20个episode后停止"""
        episodes_total = result.get("episodes_total", 0)
        timesteps_total = result.get("timesteps_total", 0)
        # 如果完成了20个episode或达到10000步，就停止
        if episodes_total >= 100 or timesteps_total >= 50000:
            print(f"停止条件满足: episodes_total={episodes_total}, timesteps_total={timesteps_total}")
            return True
        return False
    
    stop = stop_fn  # 使用自定义停止函数

    best_trial = None
    best_config = None
    
    # ===== 初始化模型管理器 =====
    model_manager = ModelManager(save_dir="./best_models", top_k=10)
    print(f"模型管理器已初始化，保存目录: ./best_models")
    
    # 用于存储训练过程中的数据
    training_data = {}
    
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

        # ===== 训练后处理：保存最好的模型 =====
        try:
            # 从结果中提取所有iteration的信息并保存模型
            if hasattr(results, 'trials') and len(results.trials) > 0:
                for trial in results.trials:
                    trial_dir = None
                    if hasattr(trial, 'logdir') and trial.logdir:
                        trial_dir = trial.logdir
                    elif hasattr(trial, 'local_dir') and trial.local_dir:
                        # 尝试从local_dir构建trial目录
                        trial_name = getattr(trial, 'trial_id', 'unknown')
                        trial_dir = os.path.join(trial.local_dir, trial_name)
                    
                    # 尝试从result.json文件读取信息（更可靠）
                    result_json_path = None
                    if trial_dir and os.path.exists(trial_dir):
                        result_json_path = os.path.join(trial_dir, "result.json")
                    
                    # 从trial.results或result.json中读取数据
                    results_data = []
                    if hasattr(trial, 'results') and trial.results:
                        results_data = trial.results
                    elif result_json_path and os.path.exists(result_json_path):
                        # 从result.json文件读取
                        try:
                            with open(result_json_path, 'r', encoding='utf-8') as f:
                                for line in f:
                                    if line.strip():
                                        try:
                                            result = json.loads(line)
                                            results_data.append(result)
                                        except json.JSONDecodeError:
                                            continue
                        except Exception as e:
                            print(f"读取result.json失败: {e}")
                    
                    # 从日志中解析episode信息（从result.json中解析）
                    episode_info_dict = parse_episode_info_from_logs(trial_dir)
                    print(f"从日志解析到的episode信息: {episode_info_dict}")
                    
                    for result in results_data:
                            iteration = result.get('training_iteration', 0)
                            if iteration == 0:
                                continue
                            
                            # 优先从解析的日志信息中获取
                            if iteration in episode_info_dict:
                                episode_info = episode_info_dict[iteration]
                                protected_facilities = episode_info.get('protected_count', 0)
                                missile_consumption = episode_info.get('missile_consumption', {'C-400': 0, 'HQ-9A': 0, 'HQ-12': 0})
                                missile_cost = episode_info.get('missile_cost', 0.0)
                            else:
                                # 如果日志中没有，尝试从custom_metrics获取
                                custom_metrics = result.get('custom_metrics', {})
                                protected_facilities = custom_metrics.get('protected_facilities_mean', 0)
                                missile_consumption = custom_metrics.get('missile_consumption_mean', {})
                                if not isinstance(missile_consumption, dict):
                                    missile_consumption = {'C-400': 0, 'HQ-9A': 0, 'HQ-12': 0}
                                missile_cost = custom_metrics.get('missile_cost_mean', 0.0)
                            
                            # 如果仍然没有获取到，尝试从result的info中获取
                            if protected_facilities == 0 and missile_cost == 0.0:
                                info_dict = result.get('info', {})
                                # 这里可以尝试从其他字段获取，但目前先使用默认值
                                pass
                            
                            # 查找对应的checkpoint路径
                            checkpoint_path = None
                            if trial_dir and os.path.exists(trial_dir):
                                # 查找checkpoint目录
                                checkpoint_pattern = os.path.join(trial_dir, f"checkpoint_{iteration}")
                                if os.path.exists(checkpoint_pattern):
                                    checkpoint_path = checkpoint_pattern
                                else:
                                    # 尝试查找最新的checkpoint
                                    checkpoint_dirs = glob.glob(os.path.join(trial_dir, "checkpoint_*"))
                                    if checkpoint_dirs:
                                        # 找到最接近当前iteration的checkpoint
                                        closest_checkpoint = None
                                        min_diff = float('inf')
                                        for cp_dir in checkpoint_dirs:
                                            try:
                                                cp_iter = int(os.path.basename(cp_dir).split('_')[1])
                                                diff = abs(cp_iter - iteration)
                                                if diff < min_diff:
                                                    min_diff = diff
                                                    closest_checkpoint = cp_dir
                                            except (ValueError, IndexError):
                                                continue
                                        if closest_checkpoint and min_diff <= 1:  # 允许1个iteration的误差
                                            checkpoint_path = closest_checkpoint
                            
                            # 如果找到了checkpoint路径，尝试保存模型
                            if checkpoint_path and os.path.exists(checkpoint_path):
                                # 确保protected_facilities是整数
                                protected_count = int(protected_facilities) if protected_facilities else 0
                                
                                # 确保missile_consumption是字典
                                if not isinstance(missile_consumption, dict):
                                    missile_consumption = {'C-400': 0, 'HQ-9A': 0, 'HQ-12': 0}
                                
                                # 尝试保存模型
                                saved = model_manager.try_save_model(
                                    protected_count=protected_count,
                                    missile_consumption=missile_consumption,
                                    iteration=iteration,
                                    checkpoint_path=checkpoint_path
                                )
                                if saved:
                                    print(f"✓ 已保存模型: 迭代={iteration}, "
                                          f"保护目标={protected_count}, "
                                          f"导弹费用={missile_cost:.1f}")
            
            print(f"\n模型保存完成，最好的模型保存在: ./best_models")
            print(f"模型信息文件: ./best_models/model_info.json")
            
        except Exception as e:
            print(f"处理训练结果时出错: {e}")
            import traceback
            traceback.print_exc()

        best_trial = results.get_best_trial('episode_reward_mean')
        best_config = results.get_best_config('episode_reward_mean')
        print(best_trial)
        print(best_config)
    except KeyboardInterrupt:
        print('\n训练被用户中断，正在保存当前模型...')
        # 模型管理器会在信号处理中自动保存
        raise
    except Exception as e:
        print(f'训练时发生异常：{str(e)}')
        # 后续放开 张志高 2021-2-16
        # reset_training_docker(training_id)
        if platform_mode == 'development':
            stop_docker(SERVER_DOCKER_DICT)
        # import traceback
        # traceback.print_exc()
    finally:
        # 确保模型管理器保存了所有信息
        try:
            model_manager._save_model_info()
        except:
            pass
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
               action_size=72,  # 已移除do-nothing机制，动作空间从73维改为72维
               obs_size=20,
               log_to_file=file_dir,
               agent_id=args.agent_id)

    print('训练结束')