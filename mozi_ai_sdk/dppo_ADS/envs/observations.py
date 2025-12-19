# 时间 ： 2020/8/31 21:10
# 作者 ： Dixit
# 文件 ： observations.py
# 项目 ： moziAIBT2
# 版权 ： 北京华戍防务技术有限公司

import numpy as np
from gym import spaces

from mozi_ai_sdk.dppo.envs import etc


class Features(object):
    def __init__(self, env, scenario, sideName):
        self.sideName = sideName
        self.side = scenario.get_side_by_name(self.sideName)
        self._env = env
        self.mozi_server = scenario.mozi_server
        
        # 初始化状态特征维度
        # 红方防空导弹系统状态：3种导弹系统 × 3个状态（可用数量、消耗数量、拦截成功率）
        # 蓝方导弹状态：2种导弹 × 3个状态（数量、距离、威胁程度）
        # 拦截状态：已拦截数量、成功拦截数量、失败拦截数量
        n_dims = 3 * 3 + 2 * 3 + 3
        
        self.action_space = self._env.action_space
        self.observation_space = spaces.Tuple([spaces.Box(0.0, float('inf'), [n_dims], dtype=np.float32),
                                               spaces.Box(0.0, 1.0, [self._env.action_space.n], dtype=np.float32)])
        
        # 初始化统计数据
        self.intercepted_count = 0
        self.intercept_success_count = 0
        self.intercept_fail_count = 0
        self.missile_consumed = {
            "远程C-400": 0,
            "近程道尔-9B": 0,
            "近程红旗-12": 0
        }
    
    def _update(self, scenario):
        """
        更新战场态势信息
        """
        self.side = scenario.get_side_by_name(self.sideName)
        self.mozi_server = scenario.mozi_server
    
    def step(self, action):
        """
        执行动作，获取新状态和奖励
        """
        scenario, mask, done = self._env.step(action)
        self._update(scenario)
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 提取特征
        obs = self._features()
        info = {}
        
        return (obs, mask), reward, done, info
    
    def reset(self):
        """
        重置环境
        """
        scenario, mask = self._env.reset()
        self._update(scenario)
        
        # 重置统计数据
        self.intercepted_count = 0
        self.intercept_success_count = 0
        self.intercept_fail_count = 0
        self.missile_consumed = {
            "远程C-400": 0,
            "近程道尔-9B": 0,
            "近程红旗-12": 0
        }
        
        # 提取特征
        obs = self._features()
        
        return (obs, mask)
    
    def _calculate_reward(self):
        """
        计算奖励
        """
        reward = 0
        
        # 获取任务对象，用于获取拦截统计信息
        task = self._env._env
        
        # 计算拦截成功奖励
        new_intercepted = len(task.intercepted_missiles)
        if new_intercepted > self.intercepted_count:
            # 计算新拦截的导弹数量
            delta = new_intercepted - self.intercepted_count
            # 假设所有新拦截的导弹都是成功拦截
            self.intercept_success_count += delta
            self.intercepted_count = new_intercepted
            
            # 根据导弹类型计算奖励
            for missile in task.blue_missiles:
                if missile["id"] in task.intercepted_missiles:
                    reward += etc.REWARD_INTERCEPT_SUCCESS[missile["type"]]
        
        # 计算拦截失败惩罚
        # 这里简化处理，实际应该根据目标是否被摧毁来计算
        
        # 计算导弹消耗惩罚
        for system in etc.RED_MISSILE_TYPES:
            if system in task.red_missile_systems:
                consumed = etc.RED_MISSILE_AVAILABLE[system] - task.red_missile_systems[system]["available"]
                delta_consumed = consumed - self.missile_consumed[system]
                if delta_consumed > 0:
                    reward -= delta_consumed * etc.PENALTY_MISSILE_COST[system]
                    self.missile_consumed[system] = consumed
        
        return reward
    
    def _features(self):
        """
        提取状态特征
        """
        features = []
        
        # 获取任务对象，用于获取状态信息
        task = self._env._env
        
        # 1. 红方防空导弹系统状态
        for system in etc.RED_MISSILE_TYPES:
            if system in task.red_missile_systems:
                # 可用数量
                available = task.red_missile_systems[system]["available"]
                # 消耗数量
                consumed = etc.RED_MISSILE_AVAILABLE[system] - available
                # 拦截成功率（使用基础成功率）
                avg_intercept_rate = 0
                count = 0
                for missile_type in etc.BLUE_MISSILE_TYPES:
                    if (system, missile_type) in etc.BASE_INTERCEPT_RATE:
                        avg_intercept_rate += etc.BASE_INTERCEPT_RATE[(system, missile_type)]
                        count += 1
                if count > 0:
                    avg_intercept_rate /= count
                
                features.extend([available, consumed, avg_intercept_rate])
            else:
                features.extend([0, 0, 0])
        
        # 2. 蓝方导弹状态
        # 统计不同类型导弹的数量和平均距离
        tomahawk_count = 0
        tomahawk_avg_dist = 0
        jassm_count = 0
        jassm_avg_dist = 0
        
        for missile in task.blue_missiles:
            if missile["id"] not in task.intercepted_missiles:
                if missile["type"] == "RGM-109E战斧巡航导弹":
                    tomahawk_count += 1
                    tomahawk_avg_dist += missile["distance"]
                elif missile["type"] == "AGM-158A联合防区外导弹":
                    jassm_count += 1
                    jassm_avg_dist += missile["distance"]
        
        # 计算平均距离
        tomahawk_avg_dist = tomahawk_avg_dist / tomahawk_count if tomahawk_count > 0 else 0
        jassm_avg_dist = jassm_avg_dist / jassm_count if jassm_count > 0 else 0
        
        # 威胁程度（距离越近威胁越大）
        tomahawk_threat = 1.0 / (tomahawk_avg_dist + 1.0) if tomahawk_count > 0 else 0
        jassm_threat = 1.0 / (jassm_avg_dist + 1.0) if jassm_count > 0 else 0
        
        features.extend([tomahawk_count, tomahawk_avg_dist, tomahawk_threat])
        features.extend([jassm_count, jassm_avg_dist, jassm_threat])
        
        # 3. 拦截状态
        features.extend([self.intercepted_count, self.intercept_success_count, self.intercept_fail_count])
        
        return np.array(features, dtype=np.float32)