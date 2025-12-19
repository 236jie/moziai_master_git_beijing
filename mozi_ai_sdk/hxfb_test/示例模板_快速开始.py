"""
强化学习智能体快速开始模板
基于 Ray RLlib + PPO 的完整示例
"""

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Discrete, Box, Dict
import argparse
from pathlib import Path
import os


# ==================== 步骤1: 定义自定义环境 ====================
class CustomRLEnv(MultiAgentEnv):
    """
    自定义强化学习环境
    继承自 MultiAgentEnv 以支持多智能体场景
    """
    
    def __init__(self, env_config):
        super().__init__()
        
        # 从配置中获取参数
        self.obs_dim = env_config.get("obs_dim", 20)
        self.action_dim = env_config.get("action_dim", 10)
        self.max_steps = env_config.get("max_steps", 200)
        self.mode = env_config.get("mode", "train")
        
        # 定义观察空间
        self.observation_space = Dict({
            "obs": Box(
                low=float("-inf"), 
                high=float("inf"), 
                shape=(self.obs_dim,),
                dtype=np.float32
            )
        })
        
        # 定义动作空间
        self.action_space = Discrete(self.action_dim)
        
        # 初始化环境状态
        self.reset()
    
    def reset(self):
        """重置环境到初始状态"""
        self.step_count = 0
        # 初始化状态（这里用随机值，实际应根据你的环境设计）
        self.state = np.random.randn(self.obs_dim).astype(np.float32)
        # 归一化到[-1, 1]范围
        self.state = np.tanh(self.state)
        
        # 返回初始观察（多智能体格式）
        return {"agent_0": {"obs": self.state}}
    
    def step(self, action_dict):
        """
        执行一步动作
        
        Args:
            action_dict: 动作字典，格式 {"agent_0": action}
        
        Returns:
            obs: 新观察
            reward: 奖励字典
            done: 完成标志字典
            info: 信息字典
        """
        # 从动作字典中提取动作（多智能体格式）
        action = action_dict.get("agent_0", 0)
        
        # 执行动作，更新状态
        # 这里是一个简单的示例：状态根据动作进行更新
        noise = np.random.randn(self.obs_dim) * 0.1
        self.state = self.state + 0.05 * (action / self.action_dim - 0.5) + noise
        self.state = np.clip(self.state, -1.0, 1.0)
        
        # 计算奖励
        # 示例：鼓励状态接近某个目标
        target = np.zeros(self.obs_dim)
        reward = -np.mean(np.abs(self.state - target))  # 负的L1距离作为奖励
        
        # 可选：添加动作惩罚
        reward -= 0.01 * abs(action - self.action_dim // 2)
        
        self.step_count += 1
        
        # 检查是否结束
        done = self.step_count >= self.max_steps
        done_dict = {"__all__": done}
        
        # 构建返回
        obs = {"agent_0": {"obs": self.state}}
        reward_dict = {"agent_0": reward}
        info = {"agent_0": {"step": self.step_count}}
        
        return obs, reward_dict, done_dict, info
    
    def get_observation(self):
        """获取当前观察（辅助方法）"""
        return {"agent_0": {"obs": self.state}}


# ==================== 步骤2: 训练脚本 ====================
def train_agent(
    training_iterations=1000,
    num_workers=2,
    obs_dim=20,
    action_dim=10,
    checkpoint_dir="./checkpoint",
    restore_path=None
):
    """
    训练强化学习智能体
    
    Args:
        training_iterations: 训练迭代次数
        num_workers: 并行worker数量
        obs_dim: 观察维度
        action_dim: 动作维度
        checkpoint_dir: 检查点保存目录
        restore_path: 恢复训练的检查点路径
    """
    
    # 定义观察和动作空间
    obs_space = Dict({
        "obs": Box(float("-inf"), float("inf"), shape=(obs_dim,))
    })
    act_space = Discrete(action_dim)
    
    # 配置训练参数
    config = {
        "env": CustomRLEnv,
        "env_config": {
            "mode": "train",
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "max_steps": 200,
        },
        "framework": "torch",
        
        # 模型配置
        "model": {
            "use_lstm": True,  # 使用LSTM处理时序
            "max_seq_len": 64,  # 序列最大长度
            "lstm_cell_size": 256,  # LSTM单元大小
            "lstm_use_prev_action_reward": True,  # 将前一步动作和奖励输入LSTM
        },
        
        # 多智能体配置
        "multiagent": {
            "agent_0": (obs_space, act_space, {"gamma": 0.99})
        },
        
        # PPO超参数
        "lambda": 0.95,  # GAE lambda
        "clip_param": 0.3,  # PPO裁剪参数
        "vf_clip_param": 10.0,  # 价值函数裁剪
        "vf_loss_coeff": 1.0,  # 价值函数损失系数
        "kl_coeff": 0.2,  # KL散度系数
        "entropy_coeff": 0.01,  # 熵系数（鼓励探索）
        "lr": 3e-4,  # 学习率
        
        # 训练参数
        "num_workers": num_workers,
        "num_envs_per_worker": 1,
        "rollout_fragment_length": 200,  # 每次rollout的步数
        "train_batch_size": 4000,  # 训练批次大小
        "sgd_minibatch_size": 128,  # SGD小批次大小
        "num_sgd_iter": 30,  # SGD迭代次数
        "batch_mode": "truncate_episodes",  # 批次模式
        
        # 其他配置
        "vf_share_layers": True,  # 价值函数和策略网络共享层
        "log_level": "WARN",
    }
    
    # 停止条件
    stop = {
        "training_iteration": training_iterations,
    }
    
    # 运行训练
    print("开始训练...")
    results = tune.run(
        "PPO",
        name="custom_rl_training",
        config=config,
        stop=stop,
        checkpoint_freq=10,  # 每10次迭代保存一次
        checkpoint_at_end=True,
        local_dir=checkpoint_dir,
        restore=restore_path,
        num_samples=1,
    )
    
    print("训练完成！")
    print(f"最佳试验: {results.get_best_trial('episode_reward_mean', 'max')}")
    
    return results


# ==================== 步骤3: 评估脚本 ====================
def evaluate_agent(
    checkpoint_path,
    num_episodes=10,
    obs_dim=20,
    action_dim=10,
    render=False
):
    """
    评估训练好的智能体
    
    Args:
        checkpoint_path: 检查点路径
        num_episodes: 评估episode数量
        obs_dim: 观察维度
        action_dim: 动作维度
        render: 是否渲染（如果环境支持）
    """
    
    # 定义空间
    obs_space = Dict({
        "obs": Box(float("-inf"), float("inf"), shape=(obs_dim,))
    })
    act_space = Discrete(action_dim)
    
    # 配置（与训练时相同）
    config = {
        "env": CustomRLEnv,
        "env_config": {
            "mode": "eval",
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "max_steps": 200,
        },
        "framework": "torch",
        "model": {
            "use_lstm": True,
            "max_seq_len": 64,
            "lstm_cell_size": 256,
            "lstm_use_prev_action_reward": True,
        },
        "multiagent": {
            "agent_0": (obs_space, act_space, {"gamma": 0.99})
        },
        "num_workers": 0,  # 评估时不需要worker
        "num_envs_per_worker": 1,
    }
    
    # 创建训练器并加载模型
    agent = PPOTrainer(env=CustomRLEnv, config=config)
    agent.restore(checkpoint_path)
    print(f"已加载模型: {checkpoint_path}")
    
    # 创建环境
    env = CustomRLEnv(config["env_config"])
    
    # 评估循环
    episode_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        
        # LSTM状态初始化
        agent_states = {}
        prev_actions = {}
        prev_rewards = {}
        
        while not done:
            # 计算动作
            agent_id = "agent_0"
            a_obs = obs[agent_id]
            
            # 使用LSTM时需要传入状态
            action, agent_states[agent_id], _ = agent.compute_action(
                a_obs,
                state=agent_states.get(agent_id),
                prev_action=prev_actions.get(agent_id),
                prev_reward=prev_rewards.get(agent_id),
                policy_id="agent_0",
                explore=False  # 评估时不探索
            )
            
            # 执行动作
            action_dict = {agent_id: action}
            obs, rewards, dones, info = env.step(action_dict)
            
            # 更新状态
            prev_actions[agent_id] = action
            prev_rewards[agent_id] = rewards[agent_id]
            total_reward += rewards[agent_id]
            done = dones["__all__"]
            step_count += 1
            
            if render:
                print(f"Step {step_count}: Action={action}, Reward={rewards[agent_id]:.4f}")
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Total Reward = {total_reward:.4f}, Steps = {step_count}")
    
    # 统计结果
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\n评估结果:")
    print(f"平均奖励: {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"最大奖励: {np.max(episode_rewards):.4f}")
    print(f"最小奖励: {np.min(episode_rewards):.4f}")
    
    agent.stop()
    return episode_rewards


# ==================== 步骤4: 主程序入口 ====================
def main():
    parser = argparse.ArgumentParser(description="强化学习智能体训练和评估")
    parser.add_argument("--mode", type=str, default="train", 
                       choices=["train", "eval"], help="运行模式")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="检查点路径（用于评估或恢复训练）")
    parser.add_argument("--iterations", type=int, default=1000,
                       help="训练迭代次数")
    parser.add_argument("--episodes", type=int, default=10,
                       help="评估episode数量")
    parser.add_argument("--obs_dim", type=int, default=20,
                       help="观察维度")
    parser.add_argument("--action_dim", type=int, default=10,
                       help="动作维度")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="并行worker数量")
    
    args = parser.parse_args()
    
    # 初始化Ray
    ray.init(local_mode=(args.num_workers == 0))
    
    if args.mode == "train":
        # 训练模式
        train_agent(
            training_iterations=args.iterations,
            num_workers=args.num_workers,
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            restore_path=args.checkpoint
        )
    elif args.mode == "eval":
        # 评估模式
        if args.checkpoint is None:
            # 自动查找最新的检查点
            checkpoint_dir = Path("./checkpoint")
            checkpoints = list(checkpoint_dir.glob("**/checkpoint-*"))
            if checkpoints:
                args.checkpoint = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
                print(f"自动找到检查点: {args.checkpoint}")
            else:
                raise ValueError("未找到检查点，请使用 --checkpoint 指定路径")
        
        evaluate_agent(
            checkpoint_path=args.checkpoint,
            num_episodes=args.episodes,
            obs_dim=args.obs_dim,
            action_dim=args.action_dim
        )
    
    ray.shutdown()


if __name__ == "__main__":
    main()


# ==================== 使用说明 ====================
"""
使用方法:

1. 训练:
   python 示例模板_快速开始.py --mode train --iterations 1000 --obs_dim 20 --action_dim 10

2. 评估:
   python 示例模板_快速开始.py --mode eval --checkpoint ./checkpoint/.../checkpoint-1000 --episodes 10

3. 自定义环境:
   - 修改 CustomRLEnv 类
   - 实现你的状态转移逻辑
   - 设计合适的奖励函数
   - 调整观察和动作空间

4. 调优建议:
   - 观察归一化很重要
   - 奖励函数设计要合理
   - 从简单环境开始测试
   - 逐步增加复杂度
   - 监控训练指标
"""



