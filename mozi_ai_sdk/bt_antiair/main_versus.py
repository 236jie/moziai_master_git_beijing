# -*- coding:utf-8 -*-
# File name : main.py
# Create date : 2020/7/20
# All rights reserved:北京华戍防务技术有限公司
# Author: Dixit
# Modified by:卡巴司机

from mozi_ai_sdk.bt_antiair.env import Environment
from mozi_ai_sdk.bt_antiair import etc
from mozi_ai_sdk.bt_antiair.bt_agent import CAgent
import os
import copy

#  设置墨子安装目录下bin目录为MOZIPATH，程序会自动启动墨子
os.environ['MOZIPATH'] = etc.MOZI_PATH


def run(env, side_name):
    """
    行为树运行的起始函数
    :param env: 墨子环境
    :param side_name: 推演方名称
    :return:
    """
    # 连接服务器，产生mozi_server
    env.start()
    # 实例化智能体
    agent = CAgent()

    # 重置函数，加载想定,拿到想定发送的数据
    env.scenario = env.reset()
    side = env.scenario.get_side_by_name(side_name)
    # 初始化行为树
    agent.init_bt(env, side_name, 0, '')
    step_count = 0
    while True:
        # 更新动作
        agent.update_bt(side_name, env.scenario)
        env.step()
        print(f"'推演步数：{step_count}")
        step_count += 1
        if env.is_done():
            print('推演已结束！')
            os.system('pause')
        else:
            pass


if __name__ == '__main__':
    env = Environment(etc.SERVER_IP, etc.SERVER_PORT, etc.PLATFORM, etc.SCENARIO_NAME, etc.SIMULATE_COMPRESSION,
                      etc.DURATION_INTERVAL, etc.SYNCHRONOUS, etc.app_mode)

    run(env, '红方')
