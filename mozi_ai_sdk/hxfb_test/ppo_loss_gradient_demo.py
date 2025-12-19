"""
PPO Loss计算和梯度更新流程演示
参考Ray RLlib的实现，简化版展示核心逻辑
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple

# 简化的数据结构
SampleBatch = namedtuple('SampleBatch', [
    'obs', 'actions', 'rewards', 'dones', 'values', 
    'old_log_probs', 'advantages', 'returns', 'state_in', 'state_out'
])


class SimplePolicy(nn.Module):
    """简化的策略网络（Actor-Critic）"""
    
    def __init__(self, obs_dim, action_dim, lstm_cell_size=256):
        super().__init__()
        
        # 共享特征提取层
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # LSTM层
        self.lstm = nn.LSTM(128, lstm_cell_size, batch_first=True)
        self.lstm_cell_size = lstm_cell_size
        
        # Actor头（策略网络）
        self.action_head = nn.Linear(lstm_cell_size, action_dim)
        
        # Critic头（价值网络）
        self.value_head = nn.Linear(lstm_cell_size, 1)
    
    def forward(self, obs, state_in=None, prev_action=None, prev_reward=None):
        """
        前向传播
        
        Args:
            obs: 观察 [batch_size, obs_dim]
            state_in: LSTM状态 (h, c) 或 None
            prev_action: 前一步动作 [batch_size, action_dim] 或 None
            prev_reward: 前一步奖励 [batch_size, 1] 或 None
        
        Returns:
            action_logits: 动作logits [batch_size, action_dim]
            value: 状态价值 [batch_size, 1]
            state_out: LSTM输出状态 (h, c)
        """
        # 1. 特征提取
        features = self.shared_net(obs)  # [batch_size, 128]
        
        # 2. 如果使用LSTM
        if state_in is not None:
            # 准备LSTM输入
            h_prev, c_prev = state_in
            features = features.unsqueeze(1)  # [batch_size, 1, 128]
            
            # LSTM处理
            lstm_out, (h_new, c_new) = self.lstm(features, (h_prev, c_prev))
            lstm_out = lstm_out.squeeze(1)  # [batch_size, lstm_cell_size]
            state_out = (h_new, c_new)
        else:
            lstm_out = features
            state_out = None
        
        # 3. Actor输出（动作概率分布）
        action_logits = self.action_head(lstm_out)  # [batch_size, action_dim]
        
        # 4. Critic输出（状态价值）
        value = self.value_head(lstm_out)  # [batch_size, 1]
        
        return action_logits, value, state_out
    
    def get_initial_state(self):
        """获取LSTM初始状态"""
        return (
            torch.zeros(1, 1, self.lstm_cell_size),  # h0
            torch.zeros(1, 1, self.lstm_cell_size)   # c0
        )


class PPOTrainer:
    """
    PPO训练器（简化版）
    演示Ray RLlib内部的loss计算和梯度更新流程
    """
    
    def __init__(self, policy, config):
        self.policy = policy
        self.config = config
        
        # 优化器
        self.optimizer = optim.Adam(
            policy.parameters(), 
            lr=config.get('lr', 3e-4)
        )
        
        # 统计信息
        self.stats = {}
    
    def compute_gae(self, rewards, values, dones, gamma=0.99, lambda_=0.98):
        """
        计算GAE（Generalized Advantage Estimation）
        
        Args:
            rewards: 奖励序列 [T]
            values: 价值估计序列 [T+1]（包含最后一步的value）
            dones: 结束标志序列 [T]
            gamma: 折扣因子
            lambda_: GAE参数（本案例中_lambda=0.98）
        
        Returns:
            advantages: 优势序列 [T]
            returns: 回报序列 [T]
        """
        T = len(rewards)
        advantages = []
        gae = 0
        
        # 从后往前计算（反向传播）
        for t in reversed(range(T)):
            if dones[t]:
                gae = 0  # Episode结束，重置GAE
            
            # TD误差
            delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            
            # GAE更新
            gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
            
            advantages.insert(0, gae)
        
        # 计算回报
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        
        return np.array(advantages), np.array(returns)
    
    def compute_loss(self, sample_batch):
        """
        计算PPO损失
        
        这是Ray RLlib内部的核心函数，计算三个损失：
        1. 策略损失（PPO裁剪）
        2. 价值损失（裁剪）
        3. 熵损失（可选）
        """
        # 1. 重新计算当前策略的动作概率和价值
        obs = torch.FloatTensor(sample_batch.obs)
        actions = torch.LongTensor(sample_batch.actions)
        
        # 处理LSTM状态
        if sample_batch.state_in is not None and len(sample_batch.state_in) > 0:
            # 使用LSTM状态
            state_in = (
                torch.FloatTensor(sample_batch.state_in[0]),
                torch.FloatTensor(sample_batch.state_in[1])
            )
        else:
            state_in = None
        
        # 前向传播
        action_logits, values, _ = self.policy(obs, state_in=state_in)
        
        # 2. 计算动作概率分布
        action_dist = torch.distributions.Categorical(logits=action_logits)
        new_action_log_probs = action_dist.log_prob(actions)
        
        # 3. 获取旧的动作概率（从sample_batch中）
        old_action_log_probs = torch.FloatTensor(sample_batch.old_log_probs)
        
        # 4. 计算重要性采样比率
        ratio = torch.exp(new_action_log_probs - old_action_log_probs)
        
        # 5. 归一化优势
        advantages = torch.FloatTensor(sample_batch.advantages)
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 6. 计算策略损失（PPO裁剪）
        clip_param = self.config.get("clip_param", 0.3)  # 本案例未设置，使用默认0.3
        
        policy_loss_1 = ratio * advantages_normalized
        policy_loss_2 = torch.clamp(
            ratio, 
            1 - clip_param, 
            1 + clip_param
        ) * advantages_normalized
        
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        
        # 7. 计算价值损失（也使用裁剪）
        vf_loss_coeff = self.config.get("vf_loss_coeff", 1.0)
        vf_clip_param = self.config.get("vf_clip_param", 10.0)  # 本案例未设置，使用默认
        
        returns = torch.FloatTensor(sample_batch.returns)
        old_values = torch.FloatTensor(sample_batch.values)
        
        vf_pred = values.squeeze()
        vf_clipped = old_values + torch.clamp(
            vf_pred - old_values,
            -vf_clip_param,
            vf_clip_param
        )
        
        vf_loss_1 = (vf_pred - returns) ** 2
        vf_loss_2 = (vf_clipped - returns) ** 2
        vf_loss = 0.5 * torch.max(vf_loss_1, vf_loss_2).mean()
        
        # 8. 计算熵损失（鼓励探索）
        entropy_coeff = self.config.get("entropy_coeff", 0.0)  # 本案例为0.0
        entropy = action_dist.entropy().mean()
        entropy_loss = -entropy_coeff * entropy
        
        # 9. 总损失
        total_loss = policy_loss + vf_loss_coeff * vf_loss + entropy_loss
        
        # 10. 计算KL散度（用于监控）
        kl = (old_action_log_probs - new_action_log_probs).mean()
        
        # 11. 计算裁剪比例（用于监控）
        clipfrac = ((ratio < 1 - clip_param) | (ratio > 1 + clip_param)).float().mean()
        
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "vf_loss": vf_loss,
            "entropy": entropy,
            "entropy_loss": entropy_loss,
            "kl": kl,
            "clipfrac": clipfrac,
            "ratio_mean": ratio.mean()
        }
    
    def update(self, sample_batch):
        """
        更新策略（apply_gradients）
        
        这是Ray RLlib内部的核心函数，执行梯度更新
        """
        # 1. 计算损失
        loss_dict = self.compute_loss(sample_batch)
        
        # 2. 反向传播
        self.optimizer.zero_grad()  # 清零梯度
        loss_dict["total_loss"].backward()  # 反向传播计算梯度
        
        # 3. 梯度裁剪（可选，防止梯度爆炸）
        max_grad_norm = self.config.get("max_grad_norm", None)
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                max_grad_norm
            )
        
        # 4. 更新参数（apply gradients）
        self.optimizer.step()  # 执行优化器步骤，更新网络参数
        
        # 5. 记录统计信息
        self.stats = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in loss_dict.items()
        }
        
        return self.stats
    
    def train_step(self, sample_batch):
        """
        完整的训练步骤
        
        模拟Ray RLlib的train()方法中的一次迭代
        """
        # 1. 多轮SGD更新（num_sgd_iter）
        num_sgd_iter = self.config.get("num_sgd_iter", 30)
        sgd_minibatch_size = self.config.get("sgd_minibatch_size", 128)
        
        all_stats = []
        
        for i in range(num_sgd_iter):
            # 2. 将数据分成小批次
            batch_size = len(sample_batch.obs)
            indices = np.random.permutation(batch_size)
            
            for start in range(0, batch_size, sgd_minibatch_size):
                end = start + sgd_minibatch_size
                batch_indices = indices[start:end]
                
                # 3. 创建小批次
                minibatch = self._create_minibatch(sample_batch, batch_indices)
                
                # 4. 更新策略
                stats = self.update(minibatch)
                all_stats.append(stats)
        
        # 5. 返回平均统计信息
        avg_stats = {
            k: np.mean([s[k] for s in all_stats])
            for k in all_stats[0].keys()
        }
        
        return avg_stats
    
    def _create_minibatch(self, sample_batch, indices):
        """创建小批次"""
        return SampleBatch(
            obs=[sample_batch.obs[i] for i in indices],
            actions=[sample_batch.actions[i] for i in indices],
            rewards=[sample_batch.rewards[i] for i in indices],
            dones=[sample_batch.dones[i] for i in indices],
            values=[sample_batch.values[i] for i in indices],
            old_log_probs=[sample_batch.old_log_probs[i] for i in indices],
            advantages=[sample_batch.advantages[i] for i in indices],
            returns=[sample_batch.returns[i] for i in indices],
            state_in=sample_batch.state_in,  # LSTM状态通常不切片
            state_out=sample_batch.state_out
        )


def demo_ppo_training():
    """
    演示PPO训练流程
    """
    print("=" * 60)
    print("PPO Loss计算和梯度更新流程演示")
    print("=" * 60)
    
    # 1. 配置（参考main_versus.py）
    config = {
        "clip_param": 0.3,
        "vf_clip_param": 10.0,
        "vf_loss_coeff": 1.0,
        "entropy_coeff": 0.0,  # 本案例为0
        "lambda": 0.98,  # GAE参数
        "gamma": 0.99,
        "lr": 3e-4,
        "num_sgd_iter": 30,
        "sgd_minibatch_size": 128,
        "max_grad_norm": 0.5
    }
    
    # 2. 创建策略网络
    obs_dim = 350  # 本案例的观察维度
    action_dim = 48  # 本案例的动作维度
    lstm_cell_size = 256  # LSTM单元大小
    
    policy = SimplePolicy(obs_dim, action_dim, lstm_cell_size)
    print(f"\n[1] 创建策略网络:")
    print(f"   观察维度: {obs_dim}")
    print(f"   动作维度: {action_dim}")
    print(f"   LSTM单元大小: {lstm_cell_size}")
    
    # 3. 创建训练器
    trainer = PPOTrainer(policy, config)
    print(f"\n[2] 创建PPO训练器")
    print(f"   优化器: Adam (lr={config['lr']})")
    
    # 4. 模拟收集的经验数据
    T = 64  # 序列长度（max_seq_len）
    print(f"\n[3] 模拟收集经验数据 (序列长度={T})")
    
    # 生成模拟数据
    obs = [np.random.randn(obs_dim) for _ in range(T)]
    actions = [np.random.randint(0, action_dim) for _ in range(T)]
    rewards = [np.random.randn() * 0.1 for _ in range(T)]
    dones = [False] * (T-1) + [True]  # 最后一步结束
    values = [np.random.randn() for _ in range(T+1)]  # 包含最后一步
    old_log_probs = [np.random.randn() * 0.1 for _ in range(T)]
    
    # 计算GAE优势
    advantages, returns = trainer.compute_gae(
        rewards, values, dones,
        gamma=config["gamma"],
        lambda_=config["lambda"]
    )
    
    print(f"   优势范围: [{advantages.min():.4f}, {advantages.max():.4f}]")
    print(f"   回报范围: [{returns.min():.4f}, {returns.max():.4f}]")
    
    # 5. 创建SampleBatch
    initial_state = policy.get_initial_state()
    sample_batch = SampleBatch(
        obs=obs,
        actions=actions,
        rewards=rewards,
        dones=dones,
        values=values[:-1],  # 不包含最后一步
        old_log_probs=old_log_probs,
        advantages=advantages.tolist(),
        returns=returns.tolist(),
        state_in=initial_state,
        state_out=None
    )
    
    # 6. 执行训练步骤
    print(f"\n[4] 执行训练步骤")
    print(f"   SGD迭代次数: {config['num_sgd_iter']}")
    print(f"   小批次大小: {config['sgd_minibatch_size']}")
    
    stats = trainer.train_step(sample_batch)
    
    # 7. 打印统计信息
    print(f"\n[5] 训练统计信息:")
    print(f"   总损失 (total_loss): {stats['total_loss']:.6f}")
    print(f"   策略损失 (policy_loss): {stats['policy_loss']:.6f}")
    print(f"   价值损失 (vf_loss): {stats['vf_loss']:.6f}")
    print(f"   熵 (entropy): {stats['entropy']:.6f}")
    print(f"   KL散度 (kl): {stats['kl']:.6f}")
    print(f"   裁剪比例 (clipfrac): {stats['clipfrac']:.4f}")
    print(f"   重要性采样比率均值 (ratio_mean): {stats['ratio_mean']:.4f}")
    
    print(f"\n[6] 梯度更新完成!")
    print(f"   网络参数已更新")
    print(f"   可以继续下一轮训练迭代")
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    demo_ppo_training()

