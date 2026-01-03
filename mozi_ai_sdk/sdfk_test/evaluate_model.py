# 时间 : 2025/12/25
# 作者 : Auto
# 文件 : evaluate_model.py
# 说明 : 加载训练好的模型，用于评估推理
# 项目 : 上海防空想定
# 版权 : 北京华戍防务技术有限公司

import argparse
import collections
import json
import os
from pathlib import Path

import ray
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.rllib.agents.ppo import PPOTrainer
from gym.spaces import Discrete, Box, Dict

from mozi_ai_sdk.sdfk_test.envs.env_sdfk import SDFKEnv

parser = argparse.ArgumentParser()
parser.add_argument("--avail_ip_port", type=str, default='127.0.0.1:6060')
parser.add_argument("--platform_mode", type=str, default='eval')
parser.add_argument("--mozi_server_path", type=str, default=r'D:\huashuanzhuang\mozilianhe\Mozi\MoziServer\bin')
# parser.add_argument("--checkpoint_path", type=str, default=r'D:\Projects\huashufangwu\python\pythonproject\moziai-master\mozi_ai_sdk\sdfk_test\best_models\rank_4_iter_1_protected_0_cost_0.0\checkpoint-1', help="模型检查点路径，如果未指定则自动查找")
# parser.add_argument("--checkpoint_path", type=str, default=r'C:\Users\81132\ray_results\test_multi_trials_1767190471\PPO_SDFKEnv_066b7_00000_0_lr=3_2025-12-31_22-14-31\checkpoint_20\checkpoint-20', help="模型检查点路径，如果未指定则自动查找")

#2026-01-02_18-48-43 版本（第二次训练的版本）
# parser.add_argument("--checkpoint_path", type=str, default=r'C:\Users\81132\ray_results\test_multi_trials_1767350923\PPO_SDFKEnv_9b65b_00000_0_lr=2_2026-01-02_18-48-43\checkpoint_16\checkpoint-16', help="模型检查点路径，如果未指定则自动查找")
# parser.add_argument("--checkpoint_path", type=str, default=r'C:\Users\81132\ray_results\test_multi_trials_1767350923\PPO_SDFKEnv_9b65b_00000_0_lr=2_2026-01-02_18-48-43\checkpoint_17\checkpoint-17', help="模型检查点路径，如果未指定则自动查找")
# parser.add_argument("--checkpoint_path", type=str, default=r'C:\Users\81132\ray_results\test_multi_trials_1767350923\PPO_SDFKEnv_9b65b_00000_0_lr=2_2026-01-02_18-48-43\checkpoint_15\checkpoint-15', help="模型检查点路径，如果未指定则自动查找")
# parser.add_argument("--checkpoint_path", type=str, default=r'C:\Users\81132\ray_results\test_multi_trials_1767350923\PPO_SDFKEnv_9b65b_00000_0_lr=2_2026-01-02_18-48-43\checkpoint_14\checkpoint-14', help="模型检查点路径，如果未指定则自动查找")

#2026-01-02_16-10-41 版本（第一次训练的版本）
# parser.add_argument("--checkpoint_path", type=str, default=r'C:\Users\81132\ray_results\test_multi_trials_1767341441\PPO_SDFKEnv_87954_00000_0_lr=2_2026-01-02_16-10-41\checkpoint_8\checkpoint-8', help="模型检查点路径，如果未指定则自动查找")
parser.add_argument("--checkpoint_path", type=str, default=r'C:\Users\81132\ray_results\test_multi_trials_1767341441\PPO_SDFKEnv_87954_00000_0_lr=2_2026-01-02_16-10-41\checkpoint_7\checkpoint-7', help="模型检查点路径，如果未指定则自动查找")
parser.add_argument("--num_episodes", type=int, default=10, help="评估轮数")


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def keep_going(episodes, num_episodes):
    """Determine whether we've collected enough data"""
    if num_episodes:
        return episodes < num_episodes
    return True


def rollout(agent,
            env_name,
            num_episodes=10,
            platform_mode=None):
    """
    执行模型评估rollout
    
    Args:
        agent: 训练好的智能体
        env_name: 环境类
        num_episodes: 评估轮数
        platform_mode: 平台模式
    """
    policy_agent_mapping = default_policy_agent_mapping

    if hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
    else:
        env = env_name()
        multiagent = False
        try:
            policy_map = {DEFAULT_POLICY_ID: agent.policy}
        except AttributeError:
            raise AttributeError(
                "Agent ({}) does not have a `policy` property! This is needed "
                "for performing (trained) agent rollouts.".format(agent))
        use_lstm = {DEFAULT_POLICY_ID: False}

    action_init = {
        p: flatten_to_single_ndarray(0)
        for p, m in policy_map.items()
    }

    episodes = 0
    episode_results = []  # 存储每轮评估结果
    
    print("=" * 80)
    print("开始模型评估")
    print("=" * 80)
    
    while keep_going(episodes, num_episodes):
        mapping_cache = {}
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        steps = 0
        
        while not done and keep_going(episodes, num_episodes):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id,
                            explore=False  # 评估时不探索
                        )
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id,
                            explore=False  # 评估时不探索
                        )
                    a_action = flatten_to_single_ndarray(a_action)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, info = env.step(action)
            
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward
            
            steps += 1
            obs = next_obs
        
        # Episode结束，提取评估指标
        protected_count = 0
        intercepted_count = 0
        missile_consumption = {'C-400': 0, 'HQ-9A': 0, 'HQ-12': 0}
        missile_cost = 0.0
        
        # 从环境的info中获取信息
        if multiagent:
            agent_info = info.get('agent_0', {}) if isinstance(info, dict) else {}
        else:
            agent_info = info.get('agent_0', {}) if isinstance(info, dict) else {}
        
        protected_count = agent_info.get('protected_facilities', 0)
        intercepted_count = agent_info.get('intercepted_missiles', 0)
        missile_consumption = agent_info.get('missile_consumption', {'C-400': 0, 'HQ-9A': 0, 'HQ-12': 0})
        missile_cost = agent_info.get('missile_cost', 0.0)
        
        # 如果从info中获取不到，尝试直接从环境对象获取
        # 注意：在rollout中，env是环境的实例，可以直接访问属性
        if protected_count == 0:
            if hasattr(env, 'protected_target'):
                protected_count = len(env.protected_target) if env.protected_target else 0
            elif hasattr(env, 'workers') and hasattr(env.workers, 'local_worker'):
                # 如果是WorkerSet，需要从local_worker获取环境
                local_env = env.workers.local_worker().env
                if hasattr(local_env, 'protected_target'):
                    protected_count = len(local_env.protected_target) if local_env.protected_target else 0
        
        if intercepted_count == 0:
            if hasattr(env, 'intercepted_missiles'):
                intercepted_count = len(env.intercepted_missiles) if env.intercepted_missiles else 0
            elif hasattr(env, 'workers') and hasattr(env.workers, 'local_worker'):
                local_env = env.workers.local_worker().env
                if hasattr(local_env, 'intercepted_missiles'):
                    intercepted_count = len(local_env.intercepted_missiles) if local_env.intercepted_missiles else 0
        
        # 如果仍然获取不到导弹消耗信息，尝试从环境对象获取
        if missile_cost == 0.0 and hasattr(env, 'missile_inventory'):
            # 尝试计算导弹消耗（需要初始库存信息）
            if hasattr(env, 'initial_missile_inventory'):
                from mozi_ai_sdk.sdfk_test.envs.env_sdfk import RED_MISSILE_COST
                missile_consumption = {
                    'C-400': env.initial_missile_inventory.get('C-400', 0) - env.missile_inventory.get('C-400', 0),
                    'HQ-9A': env.initial_missile_inventory.get('HQ-9A', 0) - env.missile_inventory.get('HQ-9A', 0),
                    'HQ-12': env.initial_missile_inventory.get('HQ-12', 0) - env.missile_inventory.get('HQ-12', 0),
                }
                # missile_cost = (
                #     RED_MISSILE_COST.get('C-400', 0) * missile_consumption['C-400'] +
                #     RED_MISSILE_COST.get('HQ-9A', 0) * missile_consumption['HQ-9A'] +
                #     RED_MISSILE_COST.get('HQ-12', 0) * missile_consumption['HQ-12']
                # )
        
        # 记录本轮结果
        episode_result = {
            'episode': episodes + 1,
            'protected_count': protected_count-1,
            'intercepted_count': intercepted_count,
            'reward': reward_total,
            'steps': steps

        }
        episode_results.append(episode_result)
        
        print(f"\n第 {episodes + 1} 轮评估结果:")
        print(f"  红方地面核心设施剩余数量: {protected_count-1}")
        print(f"  拦截成功数量: {intercepted_count}")
        print(f"  总奖励: {reward_total:.2f}")
        print(f"  步数: {steps}")
        # print(f"  导弹消耗: C-400={missile_consumption.get('C-400', 0)}, "
        #       f"HQ-9A={missile_consumption.get('HQ-9A', 0)}, "
        #       f"HQ-12={missile_consumption.get('HQ-12', 0)}")
        # print(f"  导弹消耗总费用: {missile_cost:.1f}")
        print("-" * 80)
        
        if done:
            episodes += 1
    
    # 计算统计信息
    print("\n" + "=" * 80)
    print("评估统计结果")
    print("=" * 80)
    
    if episode_results:
        avg_protected = sum(r['protected_count'] for r in episode_results) / len(episode_results)
        avg_intercepted = sum(r['intercepted_count'] for r in episode_results) / len(episode_results)
        avg_reward = sum(r['reward'] for r in episode_results) / len(episode_results)
        avg_steps = sum(r['steps'] for r in episode_results) / len(episode_results)
        # avg_missile_cost = sum(r['missile_cost'] for r in episode_results) / len(episode_results)
        #
        # # 计算导弹消耗平均值
        # avg_c400 = sum(r['missile_consumption'].get('C-400', 0) for r in episode_results) / len(episode_results)
        # avg_hq9a = sum(r['missile_consumption'].get('HQ-9A', 0) for r in episode_results) / len(episode_results)
        # avg_hq12 = sum(r['missile_consumption'].get('HQ-12', 0) for r in episode_results) / len(episode_results)
        #
        print(f"评估轮数: {len(episode_results)}")
        print(f"\n平均指标:")
        print(f"  平均红方地面核心设施剩余数量: {avg_protected:.2f}")
        print(f"  平均拦截成功数量: {avg_intercepted:.2f}")
        print(f"  平均总奖励: {avg_reward:.2f}")
        print(f"  平均步数: {avg_steps:.1f}")
        # print(f"  平均导弹消耗: C-400={avg_c400:.1f}, HQ-9A={avg_hq9a:.1f}, HQ-12={avg_hq12:.1f}")
        # print(f"  平均导弹消耗总费用: {avg_missile_cost:.1f}")
        
        # 找出最好和最差的轮次
        best_episode = max(episode_results, key=lambda x: x['protected_count'])
        worst_episode = min(episode_results, key=lambda x: x['protected_count'])
        
        print(f"\n最好轮次 (第{best_episode['episode']}轮):")
        print(f"  红方地面核心设施剩余数量: {best_episode['protected_count']}")
        print(f"  拦截成功数量: {best_episode['intercepted_count']}")
        print(f"  总奖励: {best_episode['reward']:.2f}")
        # print(f"  导弹消耗总费用: {best_episode['missile_cost']:.1f}")
        
        print(f"\n最差轮次 (第{worst_episode['episode']}轮):")
        print(f"  红方地面核心设施剩余数量: {worst_episode['protected_count']}")
        print(f"  拦截成功数量: {worst_episode['intercepted_count']}")
        print(f"  总奖励: {worst_episode['reward']:.2f}")
        # print(f"  导弹消耗总费用: {worst_episode['missile_cost']:.1f}")
        
        # 保存评估结果到文件
        result_file = "./evaluation_results.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'num_episodes': len(episode_results),
                    'avg_protected_count': avg_protected,
                    'avg_intercepted_count': avg_intercepted,
                    'avg_reward': avg_reward,
                    'avg_steps': avg_steps
                },
                'episodes': episode_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n评估结果已保存到: {result_file}")
        print("=" * 80)


def run(checkpoint=None,
        evaluate_episodes=10,
        action_size=72,  # 已移除do-nothing机制，动作空间从73维改为72维
        obs_size=20,
        avail_ip_port=None,
        Lambda=0.98,
        platform_mode='eval',
        train_config=None):
    """
    运行模型评估
    
    Args:
        checkpoint: 模型检查点路径
        evaluate_episodes: 评估轮数
        action_size: 动作空间大小
        obs_size: 观测空间大小
        avail_ip_port: 可用IP端口
        Lambda: Lambda参数
        platform_mode: 平台模式
        train_config: 训练时的配置字典（从params.json读取）
    """
    env = SDFKEnv
    act_space = Discrete(action_size)
    obs_space = Dict({"obs": Box(float("-inf"), float("inf"), shape=(obs_size,))})
    
    # 基础配置
    config = {
        "env": SDFKEnv,
        "env_config": {
            'mode': platform_mode,
            'avail_docker_ip_port': [avail_ip_port, ],
            'side_name': '红方',
            'enemy_side_name': '蓝方',
            'action_dim': action_size,
            'obs_dim': obs_size,
        },
        "framework": 'torch',
        "model": {
            "use_lstm": False,
            "max_seq_len": 64,
            "lstm_cell_size": 256,
            "lstm_use_prev_action_reward": True,
        },
        'multiagent': {
            'agent_0': (obs_space, act_space, {"gamma": 0.99}),
        },
        "vf_share_layers": True,
        "batch_mode": 'truncate_episodes',
        "num_workers": 0,
        "num_envs_per_worker": 1,
        'lambda': Lambda,
        "explore": False,  # 评估时不探索
    }
    
    # 如果提供了训练配置，合并关键参数以确保兼容性
    if train_config:
        # 合并训练时的关键参数
        train_batch_size = train_config.get('train_batch_size', -1)
        sgd_minibatch_size = train_config.get('sgd_minibatch_size', 128)
        rollout_fragment_length = train_config.get('rollout_fragment_length', 512)
        
        # 如果train_batch_size是-1，需要计算一个合理的值
        # 通常train_batch_size = rollout_fragment_length * num_workers
        # 但评估时num_workers=0，所以使用rollout_fragment_length
        if train_batch_size == -1:
            # 评估时使用rollout_fragment_length作为train_batch_size
            train_batch_size = max(rollout_fragment_length, sgd_minibatch_size)
        
        config.update({
            "lr": train_config.get('lr', 1e-4),  # 使用训练时的学习率
            "num_sgd_iter": train_config.get('num_sgd_iter', 100),
            "sgd_minibatch_size": sgd_minibatch_size,
            "rollout_fragment_length": rollout_fragment_length,
            "train_batch_size": train_batch_size,
            "vf_loss_coeff": train_config.get('vf_loss_coeff', 1.0),
            "entropy_coeff": train_config.get('entropy_coeff', 0.0),
            "kl_coeff": train_config.get('kl_coeff', 0.2),
            "vf_clip_param": train_config.get('vf_clip_param', 10.0),
            "clip_param": train_config.get('clip_param', 0.3),
        })
        # 如果训练配置中有lambda，使用训练时的值
        if 'lambda' in train_config:
            config['lambda'] = train_config['lambda']

    agent = PPOTrainer(env=env, config=config)
    
    # 尝试加载检查点
    # 如果优化器状态不匹配，Ray会抛出异常
    # 我们可以通过设置一个标志来跳过优化器加载，但Ray RLlib不直接支持
    # 所以我们需要先尝试正常加载，如果失败则使用备用方案
    try:
        agent.restore(checkpoint)
        print(f"已加载模型: {checkpoint}")
    except (ValueError, KeyError) as e:
        error_msg = str(e).lower()
        if "parameter group" in error_msg or "optimizer" in error_msg or "state dict" in error_msg:
            print(f"警告: 检测到优化器状态不匹配错误: {e}")
            print("尝试使用备用方法加载模型...")
            
            # 方法1: 尝试从checkpoint目录加载（Ray格式）
            # Ray的checkpoint通常是一个目录，包含多个文件
            checkpoint_dir = checkpoint
            if os.path.isfile(checkpoint):
                checkpoint_dir = os.path.dirname(checkpoint)
            
            # 查找checkpoint目录中的文件
            checkpoint_files = []
            if os.path.isdir(checkpoint_dir):
                for file in os.listdir(checkpoint_dir):
                    if file.startswith('checkpoint-') and not file.endswith('.tune_metadata'):
                        checkpoint_files.append(os.path.join(checkpoint_dir, file))
            
            # 如果找到了checkpoint文件，尝试使用Ray的备用加载方法
            if checkpoint_files:
                # 尝试创建一个新的agent，但使用更宽松的配置
                # 或者直接使用compute_action而不需要完整的训练器
                print("尝试重新创建agent并手动加载权重...")
                
                # 重新创建agent，使用更简单的配置（不包含可能冲突的参数）
                simple_config = config.copy()
                # 移除可能导致冲突的训练相关参数
                simple_config.pop('lr', None)
                simple_config.pop('num_sgd_iter', None)
                simple_config.pop('sgd_minibatch_size', None)
                simple_config.pop('train_batch_size', None)
                
                try:
                    agent = PPOTrainer(env=env, config=simple_config)
                    agent.restore(checkpoint)
                    print("使用简化配置成功加载模型")
                except Exception as e2:
                    print(f"简化配置加载也失败: {e2}")
                    print("请确保评估配置与训练配置一致，或使用训练时保存的params.json文件")
                    raise ValueError(f"无法加载模型检查点: {e}. 请检查配置是否与训练时一致。")
            else:
                raise ValueError(f"无法加载模型检查点: {e}. 请检查配置是否与训练时一致。")
        else:
            # 其他类型的错误，直接抛出
            raise

    print(f"已加载模型: {checkpoint}")
    
    rollout(agent, env, evaluate_episodes, platform_mode)
    agent.stop()


if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.platform_mode == 'eval':
        # 设置墨子可执行程序的路径
        os.environ['MOZIPATH'] = args.mozi_server_path
    
    # 初始化Ray
    ray.init(num_gpus=0)
    
    # 查找检查点路径
    checkpoint_path = args.checkpoint_path
    param_dir = None
    train_config = None
    
    if checkpoint_path is None:
        # 自动查找检查点
        checkpoint_base = os.path.join(Path(__file__).parent, 'checkpoint')
        checkpoint_dir = None
        
        if os.path.exists(checkpoint_base):
            for root_dir, dirs, files in os.walk(checkpoint_base):
                if 'params.json' in files:
                    param_dir = os.path.join(root_dir, 'params.json')
                for file in files:
                    if 'checkpoint-' in file:
                        checkpoint_dir = os.path.join(root_dir, file)
                        checkpoint_dir = checkpoint_dir.replace('.tune_metadata', '')
                        break
                if checkpoint_dir:
                    break
        
        if checkpoint_dir:
            checkpoint_path = checkpoint_dir
            print(f"自动找到检查点: {checkpoint_path}")
            # 如果找到了checkpoint，尝试在同一目录或父目录查找params.json
            if param_dir is None:
                checkpoint_parent = os.path.dirname(checkpoint_path)
                param_candidate = os.path.join(checkpoint_parent, 'params.json')
                if os.path.exists(param_candidate):
                    param_dir = param_candidate
        else:
            # 尝试从best_models目录查找
            best_models_dir = os.path.join(Path(__file__).parent, 'best_models')
            if os.path.exists(best_models_dir):
                # 查找最新的模型
                model_dirs = [d for d in os.listdir(best_models_dir) 
                             if os.path.isdir(os.path.join(best_models_dir, d)) and d.startswith('rank_')]
                if model_dirs:
                    # 按rank排序，取rank_1
                    model_dirs.sort()
                    latest_model_dir = os.path.join(best_models_dir, model_dirs[0])
                    # 在模型目录中查找checkpoint
                    for root_dir, dirs, files in os.walk(latest_model_dir):
                        for file in files:
                            if 'checkpoint-' in file:
                                checkpoint_path = os.path.join(root_dir, file)
                                checkpoint_path = checkpoint_path.replace('.tune_metadata', '')
                                print(f"从best_models找到检查点: {checkpoint_path}")
                                # 尝试在同一目录查找params.json
                                checkpoint_parent = os.path.dirname(checkpoint_path)
                                param_candidate = os.path.join(checkpoint_parent, 'params.json')
                                if os.path.exists(param_candidate):
                                    param_dir = param_candidate
                                break
                        if checkpoint_path:
                            break
    
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise ValueError(f'找不到检查点文件，请指定 --checkpoint_path 参数')
    
    # 读取训练配置参数（如果存在）
    Lambda = 0.98  # 默认值
    if param_dir and os.path.exists(param_dir):
        try:
            with open(param_dir, 'r', encoding='utf-8') as fp:
                train_config = json.load(fp)
                Lambda = train_config.get('lambda', 0.98)
                print(f"已读取训练配置参数: {param_dir}")
        except Exception as e:
            print(f"读取参数文件失败: {e}，使用默认值")
            train_config = None
    else:
        print("未找到params.json文件，将使用默认配置（可能导致优化器状态不匹配）")
    
    run(checkpoint=checkpoint_path,
        evaluate_episodes=args.num_episodes,
        action_size=72,  # 已移除do-nothing机制，动作空间从73维改为72维
        obs_size=20,
        avail_ip_port=args.avail_ip_port,
        Lambda=Lambda,
        platform_mode=args.platform_mode,
        train_config=train_config)

