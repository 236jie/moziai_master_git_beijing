import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.agents.dqn import DQNTrainer
import logging

# 使用 PyTorch 代替 TensorFlow，避免版本问题
config = {"framework": "torch"}

# 初始化 Ray
logging.getLogger("ray").setLevel(logging.ERROR)
ray.init(logging_level=logging.ERROR)

# 测试 PPO
try:
    ppo_trainer = PPOTrainer(config=config, env="CartPole-v1")
    print("✅ PPO Trainer 初始化成功！")
except Exception as e:
    print("❌ PPO Trainer 初始化失败:", e)

# 测试 DDPG（注意这里用 Pendulum-v0）
try:
    ddpg_trainer = DDPGTrainer(config=config, env="Pendulum-v0")
    print("✅ DDPG Trainer 初始化成功！")
except Exception as e:
    print("❌ DDPG Trainer 初始化失败:", e)

# 测试 DQN
try:
    dqn_trainer = DQNTrainer(config=config, env="CartPole-v1")
    print("✅ DQN Trainer 初始化成功！")
except Exception as e:
    print("❌ DQN Trainer 初始化失败:", e)

ray.shutdown()