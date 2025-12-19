# 时间 ： 2020/8/29 20:34
# 作者 ： Dixit
# 文件 ： tasks.py
# 项目 ： moziAIBT2
# 版权 ： 北京华戍防务技术有限公司

import datetime
import itertools
import random
import re
import uuid
from collections import namedtuple
from itertools import chain

import numpy as np

from mozi_ai_sdk.dppo_ADS.envs.spaces.mask_discrete import MaskDiscrete
from mozi_ai_sdk.dppo_ADS.envs.utils import *
from mozi_ai_sdk.dppo_ADS.envs import etc

Function = namedtuple('Function', ['type', 'function', 'is_valid'])


class Task(object):
    def __init__(self, env, scenario, sideName):
        self.scenario = scenario
        self.time = self.scenario.m_Duration.split('@')  # 想定总持续时间
        self.m_StartTime = self.scenario.m_StartTime  # 想定开始时间
        self.m_Time = self.scenario.m_Time  # 想定当前时间
        self._env = env
        self.sideName = sideName
        self.side = self.scenario.get_side_by_name(self.sideName)
        
        # 初始化蓝方导弹信息
        self.blue_missiles = []  # 存储当前探测到的蓝方导弹
        self.intercepted_missiles = set()  # 已拦截的导弹ID
        
        # 初始化红方导弹系统状态
        self.red_missile_systems = {
            "远程C-400": {
                "available": etc.RED_MISSILE_AVAILABLE["远程C-400"],
                "cost": etc.RED_MISSILE_COST["远程C-400"],
                "range": etc.RED_MISSILE_RANGE["远程C-400"]
            },
            "近程道尔-9B": {
                "available": etc.RED_MISSILE_AVAILABLE["近程道尔-9B"],
                "cost": etc.RED_MISSILE_COST["近程道尔-9B"],
                "range": etc.RED_MISSILE_RANGE["近程道尔-9B"]
            },
            "近程红旗-12": {
                "available": etc.RED_MISSILE_AVAILABLE["近程红旗-12"],
                "cost": etc.RED_MISSILE_COST["近程红旗-12"],
                "range": etc.RED_MISSILE_RANGE["近程红旗-12"]
            }
        }
        
        # 生成动作空间：针对每枚蓝方导弹，选择拦截方式
        # 动作格式：(missile_id, missile_type, defense_system, intercept_count)
        self._actions = []
        self._generate_actions()
        self.action_space = MaskDiscrete(len(self._actions))
    
    def _generate_actions(self):
        """
        生成动作空间
        动作包括：
        1. 不拦截
        2. 使用不同类型的防空导弹系统拦截
        3. 针对不同蓝方导弹类型使用不同数量的拦截导弹
        """
        # 添加不拦截动作
        self._actions.append(Function(type='donothing', function=self._ActionDoNothing(), is_valid=self._DoNothingIsValid()))
        
        # 针对战斧巡航导弹的拦截动作（需要2-3枚拦截导弹）
        for defense_system in etc.RED_MISSILE_TYPES:
            for intercept_count in [2, 3]:
                self._actions.append(Function(
                    type='intercept_tomahawk', 
                    function=self._InterceptMissileAction(defense_system, intercept_count, "RGM-109E战斧巡航导弹"),
                    is_valid=self._InterceptActionIsValid(defense_system, intercept_count)
                ))
        
        # 针对联合防区外导弹的拦截动作（需要1枚拦截导弹）
        for defense_system in etc.RED_MISSILE_TYPES:
            intercept_count = 1
            self._actions.append(Function(
                type='intercept_jassm', 
                function=self._InterceptMissileAction(defense_system, intercept_count, "AGM-158A联合防区外导弹"),
                is_valid=self._InterceptActionIsValid(defense_system, intercept_count)
            ))

    def _get_valid_action_mask(self):
        ids = [i for i, action in enumerate(self._actions) if action.is_valid()]
        mask = np.zeros(self.action_space.n)
        mask[ids] = 1
        return mask

    def _InterceptActionIsValid(self, defense_system, intercept_count):
        """
        判断拦截动作是否有效
        :param defense_system: 防空导弹系统类型
        :param intercept_count: 拦截导弹数量
        :return: 布尔值，表示动作是否有效
        """
        def is_valid():
            # 检查防空导弹系统是否有足够的导弹
            if defense_system in self.red_missile_systems:
                available_missiles = self.red_missile_systems[defense_system]["available"]
                return available_missiles >= intercept_count
            return False
        return is_valid

    def _update(self, scenario):
        """
        更新战场态势信息
        """
        self.scenario = scenario
        self.side = self.scenario.get_side_by_name(self.sideName)
        self.m_StartTime = self.scenario.m_StartTime  # 想定开始时间
        self.m_Time = self.scenario.m_Time  # 想定当前时间
        
        # 更新探测到的蓝方导弹
        self._update_blue_missiles()
        
        # 更新红方导弹系统状态
        self._update_red_missile_systems()

    def _update_blue_missiles(self):
        """
        更新当前探测到的蓝方导弹
        """
        # 从contacts中获取导弹目标
        self.blue_missiles = []
        for k, v in self.side.contacts.items():
            if v.m_ContactType == 1:  # 导弹类型
                # 根据导弹属性判断是战斧巡航导弹还是联合防区外导弹
                missile_type = "RGM-109E战斧巡航导弹"  # 默认
                if "JASSM" in v.strName or "AGM-158" in v.strName:
                    missile_type = "AGM-158A联合防区外导弹"
                
                self.blue_missiles.append({
                    "id": k,
                    "type": missile_type,
                    "latitude": v.dLatitude,
                    "longitude": v.dLongitude,
                    "distance": v.fDistance
                })

    def _update_red_missile_systems(self):
        """
        更新红方导弹系统状态
        这里简化处理，实际应该从平台获取真实状态
        """
        pass

    def step(self, action):
        """
        执行动作，推进仿真
        :param action: 选择的动作索引
        :return: scenario, mask, done
        """
        # 执行动作
        action_obj = self._actions[action]
        if action_obj.type != 'donothing':
            # 执行拦截动作
            action_obj.function()
        
        print('action:', action)
        scenario = self._env.step()  # 推进仿真
        self._update(scenario)
        mask = self._get_valid_action_mask()
        done = self._is_done()
        
        return scenario, mask, done

    def reset(self):
        """
        重置环境
        """
        scenario = self._env.reset(self.sideName)
        self._update(scenario)
        mask = self._get_valid_action_mask()
        return scenario, mask

    def _InterceptMissileAction(self, defense_system, intercept_count, blue_missile_type):
        """
        拦截导弹动作
        :param defense_system: 防空导弹系统类型
        :param intercept_count: 拦截导弹数量
        :param blue_missile_type: 蓝方导弹类型
        :return: 动作函数
        """
        def act():
            # 选择要拦截的蓝方导弹
            target_missiles = [missile for missile in self.blue_missiles 
                              if missile["type"] == blue_missile_type 
                              and missile["id"] not in self.intercepted_missiles]
            
            if not target_missiles:
                return
            
            # 选择最近的导弹进行拦截
            target_missile = min(target_missiles, key=lambda x: x["distance"])
            
            # 计算拦截成功率
            intercept_rate = etc.BASE_INTERCEPT_RATE[(defense_system, blue_missile_type)]
            
            # 根据拦截导弹数量计算实际拦截成功率
            for i in range(intercept_count - 1):
                if i < len(etc.INTERCEPT_RATE_INCREASE[defense_system]):
                    intercept_rate += etc.INTERCEPT_RATE_INCREASE[defense_system][i]
            intercept_rate = min(intercept_rate, 0.99)  # 最大成功率限制为99%
            
            # 执行拦截
            if random.random() < intercept_rate:
                # 拦截成功
                self.intercepted_missiles.add(target_missile["id"])
                print(f"成功拦截导弹: {target_missile['id']}, 类型: {blue_missile_type}, 使用: {defense_system}, 数量: {intercept_count}")
            else:
                # 拦截失败
                print(f"拦截失败: {target_missile['id']}, 类型: {blue_missile_type}, 使用: {defense_system}, 数量: {intercept_count}")
            
            # 消耗红方导弹
            if defense_system in self.red_missile_systems:
                self.red_missile_systems[defense_system]["available"] -= intercept_count
            
        return act

    def _is_done(self):
        """
        判断推演是否结束
        """
        # 检查推演是否结束标记
        response_dic = self.scenario.get_responses()
        for _, v in response_dic.items():
            if v.Type == 'EndOfDeduction':
                print('打印出标记：EndOfDeduction')
                return True
        return False

    def _DoNothingIsValid(self):
        """
        不执行动作总是有效的
        """
        def is_valid():
            return True
        return is_valid

    def _ActionDoNothing(self):
        """
        不执行任何动作
        """
        def act():
            pass
        return act