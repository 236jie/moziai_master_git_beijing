# -*- coding: utf-8 -*-
#####################################


import torch
import os

app_abspath = os.path.dirname(__file__)
app_abspath = os.path.dirname(app_abspath)
USE_CUDA = False
device = torch.device("cuda" if USE_CUDA else "cpu")


#######################
SERVER_IP = "127.0.0.1"
SERVER_PORT = "6060"
SERVER_PLAT = "windows"                 # windows linux
SCENARIO_NAME = "Sea_Confrontation2.scen"    # 1v1对抗设定
SIMULATE_COMPRESSION = 4                #推演档位
SYNCHRONOUS = True  # True同步, False异步

target_radius = 50000.0
target_name = "“阿利伯克”级导弹驱逐舰"

# task_end_point = {"latitude": 22.978889, "longitude": 118.178611}
TRANS_DATA = True
control_noise = True
#######################
# app_mode:
# 1--local windows train mode
# 2--local linux train mode
# 3--remote windows evaluate mode
# 4--local windows evaluate mode
app_mode = 1
#######################
MAX_EPISODES = 5000  # 一共训练多少轮
MAX_BUFFER = 10000
MIN_BUFFER = 500
MAX_STEPS = 100  # 一共做多少次决策
DURATION_INTERVAL = 30  # 仿真时间多长做一次决策。（单位：秒）如果为1，会导致一直在转向
#######################

#######################
TMP_PATH = "%s/%s/tmp" % (app_abspath, SCENARIO_NAME)
OUTPUT_PATH = "%s/output" % app_abspath  # 多了一层目录

MODELS_PATH = "%s/Models/" % OUTPUT_PATH  # 模型输出路径
#######################
