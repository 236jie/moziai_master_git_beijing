# LSTM在本案例中的使用详解

## 一、LSTM配置参数解析

### 1.1 配置代码位置
```python:61-67:main_versus.py
"model": {"use_lstm": True,
          # "custom_model": "mask_model",
          "max_seq_len": 64,
          # Size of the LSTM cell.
          "lstm_cell_size": 256,
          # Whether to feed a_{t-1}, r_{t-1} to LSTM.
          "lstm_use_prev_action_reward": True,
          },
```

### 1.2 参数详解

| 参数 | 值 | 作用说明 |
|------|-----|---------|
| `use_lstm` | `True` | **启用LSTM层**，使网络能够处理序列信息 |
| `max_seq_len` | `64` | **序列最大长度**，LSTM处理的最大时间步数 |
| `lstm_cell_size` | `256` | **LSTM单元大小**，隐藏状态和细胞状态的维度 |
| `lstm_use_prev_action_reward` | `True` | **将前一步动作和奖励作为输入**，增强时序信息 |

### 1.3 为什么需要LSTM？

在这个军事仿真环境中：
- **部分可观测性**: 当前观察(350维)可能不包含完整信息
- **时序依赖**: 当前决策需要依赖历史状态
- **长期记忆**: 需要记住之前的策略和结果
- **动作序列**: 动作之间存在关联性（如连续攻击同一目标）

## 二、LSTM在网络架构中的位置

### 2.1 网络结构流程

```
输入层
  ↓
[当前观察 obs (350维)]
  ↓
[前一步动作 a_{t-1}]
  ↓
[前一步奖励 r_{t-1}]
  ↓
┌─────────────────────┐
│   共享特征提取层      │  (vf_share_layers=True)
│   (全连接层)         │
└─────────────────────┘
  ↓
┌─────────────────────┐
│   LSTM层            │  ← 这里是关键！
│   - 隐藏状态: 256维  │
│   - 细胞状态: 256维  │
│   - 处理序列信息     │
└─────────────────────┘
  ↓
┌──────────┬──────────┐
│          │          │
Actor头    │    Critic头
(策略网络) │  (价值网络)
│          │          │
└──────────┴──────────┘
  ↓          ↓
动作概率    状态价值
(48维)      (标量)
```

### 2.2 LSTM的作用机制

**LSTM内部结构**:
```
LSTM单元包含：
- 遗忘门 (Forget Gate): 决定丢弃哪些信息
- 输入门 (Input Gate): 决定存储哪些新信息
- 输出门 (Output Gate): 决定输出哪些信息
- 细胞状态 (Cell State): 长期记忆
- 隐藏状态 (Hidden State): 短期记忆
```

**在本案例中**:
- **输入**: 当前观察 + 前一步动作 + 前一步奖励
- **处理**: LSTM维护256维的隐藏状态和细胞状态
- **输出**: 更新后的隐藏状态，传递给Actor和Critic头

## 三、LSTM在训练中的使用

### 3.1 训练时的数据流

```python
# 训练时，Ray RLlib自动处理LSTM状态
# 在收集经验时：

for step in range(rollout_fragment_length):
    # 1. 获取当前观察
    obs = env.get_observation()
    
    # 2. 计算动作（LSTM状态自动管理）
    action = agent.compute_action(
        obs,
        state=lstm_state,  # LSTM状态自动传递
        prev_action=prev_action,
        prev_reward=prev_reward
    )
    
    # 3. 执行动作
    next_obs, reward, done, info = env.step(action)
    
    # 4. 存储经验（包含LSTM状态）
    experience = {
        "obs": obs,
        "action": action,
        "reward": reward,
        "next_obs": next_obs,
        "state": lstm_state,  # LSTM状态被保存
        "prev_action": prev_action,
        "prev_reward": prev_reward
    }
    
    # 5. 更新LSTM状态
    lstm_state = new_lstm_state
    prev_action = action
    prev_reward = reward
```

### 3.2 训练时的状态管理

**关键点**:
1. **序列截断**: `max_seq_len=64` 限制序列长度
2. **状态重置**: Episode结束时重置LSTM状态
3. **批次处理**: 训练时按序列批次处理，保持状态连续性

## 四、LSTM在评估中的使用（重点）

### 4.1 状态初始化

```python:123-124:main_versus.py
state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
```

**代码解析**:
- `get_initial_state()`: 获取LSTM的初始状态（通常是全零）
- `use_lstm`: 检查策略是否使用LSTM（通过状态是否为空判断）

### 4.2 Episode开始时的状态重置

```python:148-153:main_versus.py
obs = env.reset()
agent_states = DefaultMapping(
    lambda agent_id: state_init[mapping_cache[agent_id]])
prev_actions = DefaultMapping(
    lambda agent_id: action_init[mapping_cache[agent_id]])
prev_rewards = collections.defaultdict(lambda: 0.)
```

**关键操作**:
- **`agent_states`**: 存储每个智能体的LSTM状态（隐藏状态+细胞状态）
- **`prev_actions`**: 存储前一步动作（初始化为0）
- **`prev_rewards`**: 存储前一步奖励（初始化为0.0）

### 4.3 每一步的状态传递和更新

```python:164-173:main_versus.py
if p_use_lstm:
    a_action, p_state, _ = agent.compute_action(
        a_obs,
        state=agent_states[agent_id],        # ← 传入当前LSTM状态
        prev_action=prev_actions[agent_id],  # ← 传入前一步动作
        prev_reward=prev_rewards[agent_id],  # ← 传入前一步奖励
        policy_id=policy_id,
        # explore=False
    )
    agent_states[agent_id] = p_state        # ← 更新LSTM状态
```

**详细流程**:

1. **输入准备**:
   ```python
   输入 = {
       "obs": 当前观察 (350维),
       "state": agent_states[agent_id],      # LSTM隐藏状态+细胞状态
       "prev_action": prev_actions[agent_id], # 前一步动作
       "prev_reward": prev_rewards[agent_id]   # 前一步奖励
   }
   ```

2. **LSTM处理**:
   ```python
   # 在LSTM内部：
   # 1. 将 obs + prev_action + prev_reward 拼接
   # 2. 通过LSTM单元处理
   # 3. 更新隐藏状态和细胞状态
   new_hidden_state, new_cell_state = LSTM(
       input=concat(obs, prev_action, prev_reward),
       hidden_state=old_hidden_state,
       cell_state=old_cell_state
   )
   ```

3. **输出和状态更新**:
   ```python
   # compute_action返回：
   a_action = 动作 (48维离散动作)
   p_state = (new_hidden_state, new_cell_state)  # 新的LSTM状态
   
   # 更新状态供下一步使用
   agent_states[agent_id] = p_state
   ```

### 4.4 完整的状态传递循环

```python
# Episode开始
obs = env.reset()
agent_states = {初始LSTM状态}  # 全零状态
prev_actions = {初始动作}      # 0
prev_rewards = {初始奖励}      # 0.0

# Episode循环
while not done:
    # 步骤1: 计算动作（使用LSTM状态）
    action, new_lstm_state, _ = agent.compute_action(
        obs,
        state=agent_states,      # 使用当前LSTM状态
        prev_action=prev_actions,
        prev_reward=prev_rewards
    )
    
    # 步骤2: 更新LSTM状态（为下一步准备）
    agent_states = new_lstm_state
    
    # 步骤3: 执行动作
    next_obs, reward, done, info = env.step(action)
    
    # 步骤4: 保存当前动作和奖励（供下一步使用）
    prev_actions = action
    prev_rewards = reward
    
    # 步骤5: 更新观察
    obs = next_obs
```

## 五、LSTM状态的数据结构

### 5.1 状态内容

LSTM状态通常是一个**元组**，包含：
```python
lstm_state = (
    hidden_state,  # 形状: (batch_size, lstm_cell_size) = (1, 256)
    cell_state     # 形状: (batch_size, lstm_cell_size) = (1, 256)
)
```

### 5.2 状态维度

- **隐藏状态 (Hidden State)**: `[1, 256]` - 短期记忆
- **细胞状态 (Cell State)**: `[1, 256]` - 长期记忆
- **总状态大小**: `2 × 256 = 512` 个浮点数

### 5.3 状态初始化

```python
# 初始状态通常是全零
initial_state = (
    np.zeros((1, 256)),  # 隐藏状态
    np.zeros((1, 256))   # 细胞状态
)
```

## 六、为什么需要 `lstm_use_prev_action_reward=True`？

### 6.1 增强时序信息

**默认情况**（`lstm_use_prev_action_reward=False`）:
```
LSTM输入 = [当前观察 obs_t]
```

**启用后**（`lstm_use_prev_action_reward=True`）:
```
LSTM输入 = [当前观察 obs_t, 前一步动作 a_{t-1}, 前一步奖励 r_{t-1}]
```

### 6.2 优势

1. **动作连续性**: 智能体知道上一步做了什么
2. **奖励反馈**: 智能体知道上一步动作的效果
3. **策略一致性**: 帮助学习连贯的策略序列
4. **因果理解**: 理解动作-奖励的因果关系

### 6.3 在本案例中的意义

在军事仿真中：
- **战术连贯性**: 连续攻击同一目标需要记忆
- **效果评估**: 需要知道上次行动是否成功
- **策略调整**: 根据上次结果调整当前策略

## 七、关键代码片段完整解析

### 7.1 状态检测

```python:123-124:main_versus.py
state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
```

**作用**: 
- 获取每个策略的初始LSTM状态
- 通过状态是否为空判断是否使用LSTM
- `len(s) > 0` 表示有LSTM状态（元组非空）

### 7.2 状态传递的核心代码

```python:164-173:main_versus.py
if p_use_lstm:
    a_action, p_state, _ = agent.compute_action(
        a_obs,
        state=agent_states[agent_id],        # 当前LSTM状态
        prev_action=prev_actions[agent_id],  # 前一步动作
        prev_reward=prev_rewards[agent_id],  # 前一步奖励
        policy_id=policy_id,
    )
    agent_states[agent_id] = p_state        # 更新状态
```

**关键点**:
1. **`compute_action`返回三个值**:
   - `a_action`: 动作
   - `p_state`: 新的LSTM状态
   - `_`: 其他信息（未使用）

2. **状态必须更新**: `agent_states[agent_id] = p_state`
   - 如果不更新，下一步会使用旧状态
   - 导致LSTM无法正确学习时序依赖

3. **前一步信息必须传递**:
   - `prev_action` 和 `prev_reward` 是LSTM的输入
   - 这些信息帮助LSTM理解动作序列和奖励序列

### 7.3 状态重置时机

```python:148:main_versus.py
obs = env.reset()
```

**每次`env.reset()`时**:
- 环境状态重置
- **LSTM状态也应该重置**（在代码149-150行重新初始化）

**为什么需要重置？**
- 新Episode与旧Episode无关
- 保持状态独立性
- 避免状态污染

## 八、常见问题和注意事项

### 8.1 忘记更新LSTM状态

❌ **错误做法**:
```python
action, state, _ = agent.compute_action(obs, state=agent_states)
# 忘记更新 agent_states = state
```

✅ **正确做法**:
```python
action, state, _ = agent.compute_action(obs, state=agent_states)
agent_states = state  # 必须更新！
```

### 8.2 忘记传递前一步信息

❌ **错误做法**:
```python
action = agent.compute_action(obs, state=agent_states)
# 没有传递 prev_action 和 prev_reward
```

✅ **正确做法**:
```python
action, state, _ = agent.compute_action(
    obs,
    state=agent_states,
    prev_action=prev_actions,
    prev_reward=prev_rewards
)
```

### 8.3 Episode间状态未重置

❌ **错误做法**:
```python
for episode in range(num_episodes):
    obs = env.reset()
    # 忘记重置 agent_states
    while not done:
        ...
```

✅ **正确做法**:
```python
for episode in range(num_episodes):
    obs = env.reset()
    agent_states = get_initial_state()  # 重置LSTM状态
    prev_actions = initial_actions
    prev_rewards = initial_rewards
    while not done:
        ...
```

### 8.4 max_seq_len 设置

- **太小** (如16): 无法处理长序列依赖
- **太大** (如256): 增加计算和内存开销
- **合理值** (64): 平衡性能和效果

## 九、性能优化建议

### 9.1 LSTM大小选择

- **简单任务**: `lstm_cell_size=128`
- **中等任务**: `lstm_cell_size=256` (本案例)
- **复杂任务**: `lstm_cell_size=512`

### 9.2 序列长度优化

- 根据任务特点调整 `max_seq_len`
- 如果Episode平均长度是50步，设置64是合理的
- 如果Episode很长（>200步），考虑增加

### 9.3 内存管理

LSTM状态占用内存：
```
每个智能体: 2 × 256 × 4 bytes = 2KB (float32)
100个智能体: 200KB
```

在分布式训练中，注意worker数量对内存的影响。

## 十、总结

### 10.1 LSTM在本案例中的关键作用

1. **处理时序依赖**: 记住历史状态和动作
2. **部分可观测**: 通过历史信息补充当前观察
3. **策略连贯性**: 学习连贯的动作序列
4. **长期记忆**: 通过细胞状态保持长期信息

### 10.2 使用LSTM的完整流程

```
配置LSTM → 初始化状态 → 传递状态 → 更新状态 → 重置状态
   ↓           ↓            ↓          ↓          ↓
训练/评估    Episode开始   每步计算   每步更新   Episode结束
```

### 10.3 关键代码位置

- **配置**: `main_versus.py:61-67`
- **状态初始化**: `main_versus.py:123-124, 149-153`
- **状态传递**: `main_versus.py:164-173`
- **状态更新**: `main_versus.py:173`

---

**提示**: 理解LSTM状态管理是使用RLlib进行时序强化学习的关键。确保正确传递和更新状态，否则智能体无法学习有效的时序策略。



