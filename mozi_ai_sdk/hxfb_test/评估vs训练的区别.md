# 评估 vs 训练：Loss和梯度的区别

## 一、核心结论

**评估过程不需要计算loss和梯度！**

评估只是使用已训练好的模型进行前向传播，获取动作并执行，不涉及任何参数更新。

## 二、代码对比

### 2.1 评估模式（main_versus.py）

```python
# 第82-84行
agent = PPOTrainer(env=env, config=config)
agent.restore(checkpoint)  # 只加载模型，不训练
rollout(agent, env, evaluate_episodes, platform_mode)  # 只评估
```

**rollout函数中的关键代码**（第165行）：
```python
# 只调用 compute_action，不调用 train
a_action, p_state, _ = agent.compute_action(
    a_obs,
    state=agent_states[agent_id],
    prev_action=prev_actions[agent_id],
    prev_reward=prev_rewards[agent_id],
    policy_id=policy_id,
    # explore=False  # 评估时不探索
)
```

**`compute_action()` 做了什么**：
1. ✅ **前向传播**：输入观察 → 网络 → 输出动作
2. ✅ **LSTM状态管理**：更新和传递LSTM状态
3. ❌ **不计算loss**：没有计算任何损失函数
4. ❌ **不计算梯度**：没有调用 `backward()`
5. ❌ **不更新参数**：没有调用 `optimizer.step()`

### 2.2 训练模式（main_train.py）

```python
# 训练时使用 tune.run()
results = tune.run(
    "PPO",  # 或 "DDPPO"
    config=config,
    ...
)

# 内部会调用 agent.train()，包含：
# 1. 收集经验
# 2. 计算GAE优势
# 3. 计算loss
# 4. 反向传播
# 5. 更新参数
```

## 三、详细对比

### 3.1 评估流程（Evaluation）

```
加载模型 (restore)
    ↓
┌─────────────────────────────┐
│  评估循环 (rollout)          │
│                             │
│  1. 环境重置                 │
│     obs = env.reset()       │
│                             │
│  2. 计算动作（前向传播）     │
│     action = agent.compute_action(obs) │
│     - 输入: obs, LSTM状态   │
│     - 网络: 前向传播        │
│     - 输出: action          │
│     ❌ 不计算loss           │
│     ❌ 不计算梯度           │
│                             │
│  3. 执行动作                 │
│     obs, reward, done = env.step(action) │
│                             │
│  4. 记录奖励                 │
│     total_reward += reward  │
│                             │
│  5. 重复直到episode结束     │
└─────────────────────────────┘
    ↓
打印评估结果
```

### 3.2 训练流程（Training）

```
初始化模型
    ↓
┌─────────────────────────────┐
│  训练循环 (train)            │
│                             │
│  1. 数据收集                 │
│     sample_batch = workers.sample() │
│                             │
│  2. 计算优势                 │
│     advantages = compute_gae(...) │
│                             │
│  3. 计算Loss ✅              │
│     loss = compute_ppo_loss(...) │
│     - 策略损失               │
│     - 价值损失               │
│     - 熵损失                 │
│                             │
│  4. 反向传播 ✅              │
│     loss.backward()         │
│                             │
│  5. 计算梯度 ✅              │
│     gradients = ...          │
│                             │
│  6. 更新参数 ✅              │
│     optimizer.step()         │
│                             │
│  7. 同步权重                 │
│     workers.set_weights(...) │
└─────────────────────────────┘
    ↓
继续下一轮迭代
```

## 四、compute_action() 内部实现（简化）

```python
def compute_action(self, obs, state=None, prev_action=None, prev_reward=None):
    """
    评估时使用：只做前向传播，不计算loss和梯度
    """
    # 1. 设置评估模式（关闭dropout等）
    self.policy.eval()
    
    with torch.no_grad():  # ← 关键：不计算梯度！
        # 2. 前向传播
        action_logits, value, new_state = self.policy(
            obs, 
            state=state,
            prev_action=prev_action,
            prev_reward=prev_reward
        )
        
        # 3. 采样动作（评估时通常不探索）
        action_dist = Categorical(logits=action_logits)
        action = action_dist.sample()  # 或 action_dist.mode()（确定性）
    
    return action, new_state, {}
    # 返回：动作、新LSTM状态、空字典（不包含loss信息）
```

**关键点**：
- `torch.no_grad()`：禁用梯度计算，节省内存和计算
- `policy.eval()`：设置为评估模式
- 只返回动作，不返回loss

## 五、train() 内部实现（简化）

```python
def train(self):
    """
    训练时使用：计算loss、梯度并更新参数
    """
    # 1. 设置训练模式
    self.policy.train()
    
    # 2. 收集经验
    sample_batch = self.workers.sample()
    
    # 3. 计算优势
    advantages = self.compute_gae(sample_batch)
    
    # 4. 计算Loss ✅
    loss_dict = self.compute_ppo_loss(sample_batch, advantages)
    
    # 5. 反向传播 ✅
    self.optimizer.zero_grad()
    loss_dict["total_loss"].backward()  # 计算梯度
    
    # 6. 更新参数 ✅
    self.optimizer.step()
    
    return loss_dict
```

## 六、为什么评估不需要loss和梯度？

### 6.1 评估的目的

- **测试模型性能**：看训练好的模型表现如何
- **不改变模型**：保持模型参数不变
- **只做推理**：输入观察，输出动作

### 6.2 计算loss和梯度的目的

- **更新模型参数**：通过梯度下降优化策略
- **学习更好的策略**：改进模型性能
- **需要大量计算**：反向传播、梯度计算等

### 6.3 评估时计算loss和梯度的问题

1. **浪费计算资源**：评估时不需要更新参数
2. **占用内存**：梯度需要存储中间结果
3. **可能改变模型**：如果误操作可能更新参数

## 七、实际代码验证

### 7.1 main_versus.py（评估）

```python
# 第82-84行
agent = PPOTrainer(env=env, config=config)
agent.restore(checkpoint)  # 加载模型
rollout(agent, env, evaluate_episodes, platform_mode)  # 只评估

# 在rollout中：
# ✅ 调用 agent.compute_action() - 只前向传播
# ❌ 不调用 agent.train() - 不训练
# ❌ 不计算loss
# ❌ 不计算梯度
```

### 7.2 main_train.py（训练）

```python
# 使用 tune.run() 进行训练
results = tune.run(
    "PPO",
    config=config,
    ...
)

# 内部会：
# ✅ 调用 agent.train() - 完整训练流程
# ✅ 计算loss
# ✅ 计算梯度
# ✅ 更新参数
```

## 八、总结

| 项目 | 评估（Evaluation） | 训练（Training） |
|------|-------------------|-----------------|
| **目的** | 测试模型性能 | 优化模型参数 |
| **调用方法** | `compute_action()` | `train()` |
| **前向传播** | ✅ 需要 | ✅ 需要 |
| **计算Loss** | ❌ **不需要** | ✅ 需要 |
| **计算梯度** | ❌ **不需要** | ✅ 需要 |
| **更新参数** | ❌ **不需要** | ✅ 需要 |
| **torch.no_grad()** | ✅ 使用 | ❌ 不使用 |
| **内存占用** | 低 | 高 |
| **计算开销** | 低 | 高 |

## 九、关键要点

1. **评估 = 推理**：只做前向传播，获取动作
2. **训练 = 学习**：前向传播 + 反向传播 + 参数更新
3. **评估时使用 `torch.no_grad()`**：禁用梯度计算，节省资源
4. **评估时模型参数不变**：只测试，不修改
5. **main_versus.py 是评估脚本**：不进行训练

---

**结论**：评估过程**不需要**计算loss和梯度，只需要前向传播获取动作即可。这是评估和训练的根本区别。

