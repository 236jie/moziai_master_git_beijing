# 时间 ： 2020/9/8 21:30
# 作者 ： Dixit
# 文件 ： etc.py
# 项目 ： moziAIBT2
# 版权 ： 北京华戍防务技术有限公司


import os
import sys

APP_ABSPATH = os.path.dirname(__file__)

#######################
SERVER_IP = "127.0.0.1"
SERVER_PORT = "6060"
PLATFORM = 'windows' if sys.platform=='win32' else sys.platform
# SCENARIO_NAME = "bt_test.scen"
SCENARIO_NAME = "首都防空.scen"
# SCENARIO_NAME = "hxfb"
SIMULATE_COMPRESSION = 3
DURATION_INTERVAL = 5
SYNCHRONOUS = True
#######################
# app_mode:
# 1--local windows train mode
# 2--local linux train mode
# 3--remote windows evaluate mode
# 4--local windows evaluate mode
app_mode = 1
#######################
MAX_EPISODES = 5000
MAX_BUFFER = 1000000
MAX_STEPS = 30
#######################

# 蓝方导弹配置
BLUE_MISSILE_TYPES = ["RGM-109E战斧巡航导弹", "AGM-158A联合防区外导弹"]
BLUE_MISSILE_WEIGHTS = {"RGM-109E战斧巡航导弹": 0.4, "AGM-158A联合防区外导弹": 0.6}
BLUE_MISSILE_COUNT = {"RGM-109E战斧巡航导弹": 64, "AGM-158A联合防区外导弹": 48}

# 红方导弹配置
RED_MISSILE_TYPES = ["远程C-400", "近程道尔-9B", "近程红旗-12"]
RED_MISSILE_AVAILABLE = {"远程C-400": 160, "近程道尔-9B": 96, "近程红旗-12": 128}
RED_MISSILE_COST = {"远程C-400": 20, "近程道尔-9B": 15, "近程红旗-12": 10}
RED_MISSILE_RANGE = {"远程C-400": 380, "近程道尔-9B": 140, "近程红旗-12": 55}

# 拦截成功率模型配置
BASE_INTERCEPT_RATE = {
    ("远程C-400", "RGM-109E战斧巡航导弹"): 0.8,
    ("远程C-400", "AGM-158A联合防区外导弹"): 0.6,
    ("近程道尔-9B", "RGM-109E战斧巡航导弹"): 0.6,
    ("近程道尔-9B", "AGM-158A联合防区外导弹"): 0.4,
    ("近程红旗-12", "RGM-109E战斧巡航导弹"): 0.4,
    ("近程红旗-12", "AGM-158A联合防区外导弹"): 0.2
}

INTERCEPT_RATE_INCREASE = {
    "远程C-400": [0.2, 0.15, 0.1, 0.5, 0.5],
    "近程道尔-9B": [0.18, 0.12, 0.6, 0.3, 0.3],
    "近程红旗-12": [0.15, 0.8, 0.4, 0.1, 0.1]
}

MAX_INTERCEPT_MISSILES = 10

# 奖励配置
REWARD_INTERCEPT_SUCCESS = {
    "RGM-109E战斧巡航导弹": 200,
    "AGM-158A联合防区外导弹": 150
}

REWARD_INTERCEPT_FAIL = {
    "RGM-109E战斧巡航导弹": 80,
    "AGM-158A联合防区外导弹": 50
}

PENALTY_MISSILE_COST = {
    "远程C-400": 20,
    "近程道尔-9B": 15,
    "近程红旗-12": 10
}

PENALTY_TARGET_DESTROYED = -9999

# PPO算法配置
PPO_EPOCHS = 10
PPO_CLIP_PARAM = 0.2
PPO_LR = 3e-4
PPO_BATCH_SIZE = 64
PPO_GAMMA = 0.99
PPO_LAM = 0.95

# 训练配置
EPISODE_LENGTH = 1000
TOTAL_EPISODES = 1000
STATE_DIM = 100  # 需要根据实际特征维度调整
ACTION_DIM = 10  # 需要根据实际动作维度调整
EVAL_FREQUENCY = 100

# 输出配置
SAVE_MODEL_PATH = "./bin/checkpoints/"
LOG_PATH = "./bin/logs/"
OUTPUT_INTERCEPT_SCHEME = True

#######################
TMP_PATH = "%s/%s/tmp" % (APP_ABSPATH, SCENARIO_NAME)
OUTPUT_PATH = "%s/%s/output" % (APP_ABSPATH, SCENARIO_NAME)

CMD_LUA = "%s/cmd_lua" % TMP_PATH
PATH_CSV = "%s/path_csv" % OUTPUT_PATH
MODELS_PATH = "%s/Models/" % OUTPUT_PATH
EPOCH_FILE = "%s/epochs.txt" % OUTPUT_PATH
#######################

MOZIPATH = 'E:\\Mozi_C#\\Mozi\\MoziServer0412\\bin'
