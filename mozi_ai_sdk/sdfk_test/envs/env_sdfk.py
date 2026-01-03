#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块功能：提供三防空导弹体系的墨子仿真环境封装，支持多智能体训练与对战。
包含类：SDFKEnv（继承MultiAgentEnv，负责环境交互与状态特征构造）
依赖：mozi_ai_sdk（环境交互）、ray.rllib（多智能体接口）、numpy、zmq（docker重启通信）
适用场景：训练/评估红方防空拦截策略，支持train/development/versus/eval多种模式(训练、进化、竞争、评估)。
"""
"""
Created on Thu Oct 15 10:33:22 2020

@author: dixit
"""

import random  # 随机数工具，用于动作采样或随机策略
import itertools  # 迭代器工具，便于组合/排列操作
import uuid  # 生成唯一ID，可用于任务标识
import numpy as np  # 数值计算库，构造观测向量等
from collections import namedtuple  # 轻量结构体，用于封装函数表
from itertools import chain  # 迭代器拼接
from mozi_simu_sdk.mssnpatrol import CPatrolMission  # 巡逻任务接口，可能用于扩展
from mozi_simu_sdk.mssnstrike import CStrikeMission  # 打击任务接口
# from mozi_ai_sdk.test.dppo.envs.common.utils import *
from mozi_ai_sdk.sdfk_test.envs.common.utils import *  # 项目通用工具函数
from mozi_utils.geo import get_two_point_distance  # 计算经纬度距离
from mozi_ai_sdk.sdfk_test.envs.env import Environment  # 墨子环境封装
from mozi_ai_sdk.sdfk_test.envs import etc  # 想定配置常量

from ray.rllib.env.multi_agent_env import MultiAgentEnv  # RLlib多智能体环境基类
from gym.spaces import Discrete, Box, Dict  # Gym空间定义
from ray.remote_handle_docker import restart_mozi_container  # docker重启工具

import sys  # 系统级操作，例如退出
import re  # 正则表达式，用于武器ID解析
import zmq  # 进程间通信，向调度端发重启命令
import time  # 时间戳及睡眠

#在新线程中显示警告窗口
import tkinter as tk
from tkinter import messagebox
import threading

import ctypes
import platform

# zmq init
zmq_context = zmq.Context()  # 初始化ZMQ上下文，用于创建通信套接字
# ray request port
restart_requestor = zmq_context.socket(zmq.REQ)  # 请求型socket，用于发送docker重启指令
Function = namedtuple('Function', ['type', 'function'])  # 封装动作函数表的结构体
FEATS_MAX_LEN = 350  # 观测特征最大长度预留（兼容旧代码）
MAX_DOCKER_RETRIES = 3  # docker重启最大尝试次数
# 红方单元配置
RED_MISSILE_NUMBERS = [("C-400", 2), ("HQ-9A", 3), ("HQ-12", 4)]  # 可用防空营数量
RED_UNITS = ["C-400", "HQ-9A", "HQ-12"]  # 防空单元类型列表
# 2104是武器的dbid，HQ-12初始可用数量12，打完后剩下的12枚导弹重装载需要2分钟

# RED_MISSILE_INFO_MAP = {"C-400": (2104, "萨姆-21B型", 48), "HQ-9A": (1225,"HQ-9A", 32), "HQ-12": (123,"HQ-12", 36)}  # 武器DBID及理论装弹数
# RED_MISSILE_AVAILABLE = {"C-400": 48, "HQ-9A": 32, "HQ-12": 36}  # HQ-12初始可用数量12，打完后剩下的12枚导弹重装载需要2分钟

# RED_MISSILE_INFO_MAP = {"C-400": (2104, "萨姆-21B型", 80), "HQ-9A": (1225,"HQ-9A", 64), "HQ-12": (123,"HQ-12", 36)}  # 武器DBID及理论装弹数
RED_MISSILE_INFO_MAP = {"C-400": (2104, "萨姆-21B型", 80), "HQ-9A": (1225,"HQ-9A", 64), "HQ-12": (2000002,"HQ-12", 14)}  # 武器DBID及理论装弹数
RED_MISSILE_AVAILABLE = {"C-400": 80, "HQ-9A": 64, "HQ-12": 14}  # HQ-12初始可用数量12，打完后剩下的12枚导弹重装载需要2分钟

RED_MISSILE_COST = {"C-400": 30, "HQ-9A": 20, "HQ-12": 10}  # 每种类型导弹的价值
# RED_MISSILE_RANGE = {"C-400": 380, "HQ-9A": 140, "HQ-12": 55}  #在本想定推演中，蓝方导弹在70km才能被红方导弹探测到
RED_MISSILE_RANGE = {"C-400": 180, "HQ-9A": 100, "HQ-12": 70}  # 在本想定推演中，蓝方导弹在70km才能被红方导弹探测到，C-400射程已改为180km


def show_warning():
    """在新线程中显示警告窗口"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    messagebox.showwarning("警告", "平台类型为HQ-12，但是返回为空！")
    root.destroy()

def restart_container(schedule_addr, schedule_port, _training_id, docker_ip_port):
    """
    控制进程层面重启训练docker容器，避免墨子仿真长期运行导致异常。

    Args:
        schedule_addr: 调度服务器地址，用于发送重启请求。
        schedule_port: 调度服务器端口。
        _training_id: 当前训练任务的唯一标识。
        docker_ip_port: 目标docker实例的IP:PORT字符串。

    Returns:
        str: 成功重启后返回的docker_ip_port。

    Raises:
        SystemExit: 当重启失败或返回信息异常时退出进程。
    """
    # 训练5轮后，重启docker
    try:
        message = {}
        message['zmq_command'] = 'restart_training_container'
        message['docker_ip_port'] = docker_ip_port
        message['training_id'] = _training_id
        restart_requestor.connect("tcp://%s:%s" % (str(schedule_addr), str(schedule_port)))
        restart_requestor.send_pyobj(message)
        recv_msg = restart_requestor.recv_pyobj()
        assert type(recv_msg) == str
        if 'OK' in recv_msg:
            pass
        else:
            sys.exit(1)
        return docker_ip_port
    except Exception:
        print('fail restart mozi docker!')
        sys.exit(1)


def _manual_attack(unit, target, missile_count, weapon_dbid, platform_type, step=None):
    """
    调用墨子平台的手动打击接口（人工分配调用）

    Args:
        unit: 执行拦截的防空单元
        target: 目标导弹
        missile_count: 发射导弹数量
        weapon_dbid: 武器DBID(平台对应的DBID)
        platform_type: 平台类型（C-400/HQ-9A/HQ-12）
        step: 当前步骤数（可选，用于输出日志）

    Returns:
        bool: 是否成功发送指令
    """
    try:
        result = unit.manual_attack(target.strGuid, weapon_dbid, missile_count)

        if '成功' in result:
            # 输出人工分配调用信息
            step_info = f"[步骤{step}] " if step is not None else ""
            print(f"{step_info}人工分配调用{platform_type}发射{missile_count}枚导弹")
            return True
        else:
            print(f"手动打击指令执行失败：{result}")
            return False

    except Exception as e:
        print(f"执行手动打击时发生异常：{e}")
        return False


class SDFKEnv(MultiAgentEnv):
    """
    红方防空多智能体环境，封装墨子仿真接口并提供RLlib兼容的step/reset。

    属性：
        env_config: 环境配置字典（模式、侧别、端口、奖励塑形等）。
        env: Environment实例，负责与墨子仿真交互。
        scenario: 当前想定对象，包含时间、态势等信息。
        side/enemy_side: 红蓝双方势力对象。
        reward_accum/temp_reward: 奖励累积与临时塑形奖励。
        missile_inventory: 三类防空单元的剩余弹药估计。
        # 注意：冷却机制已完全移除
        action_space/observation_space: RL动作、观测空间定义。

    主要方法：
        step/reset: RL环境标准接口。
        _parse_action: 将MultiDiscrete动作解析为拦截计划。
        _execute_engagement_plan: 调用手动打击并记录奖励。
        _generate_features: 构造训练用观测特征。
        _is_done/_get_win_score: 终止条件与奖励设计。
    使用场景：训练或评估红方防空体系的拦截策略。
    """
    def __init__(self, env_config):
        """
        初始化环境并完成首次reset，从仿真获取初始态势。

        Args:
            env_config: 运行模式、端口、侧别、奖励参数等配置。
        """
        self.steps = None
        # 初始化累积奖励为0，避免前一局遗留None带来的异常
        self.reward_accum = 0.0
        self.env_config = env_config
        self.reset_nums = 0
        self._get_env()  # 按配置创建墨子环境实例并启动docker
        self.side_name = env_config['side_name']
        print('开始 reset!!!')
        self.scenario = self.env.reset(self.side_name)  # 首次reset以获取初始态势
        print('结束 reset!!!')

        self.time = self.scenario.m_Duration.split('@')  # 想定总持续时间
        self.m_StartTime = self.scenario.m_StartTime  # 想定开始时间
        self.m_Time = self.scenario.m_Time  # 想定当前时间

        self.side = self.scenario.get_side_by_name(self.side_name)
        self.enemy_side = self.scenario.get_side_by_name(env_config['enemy_side_name'])
        self.reward = float(self.side.iTotalScore) / 4067
        self.temp_reward = 0
        # 追踪探测到的目标集合（在reset时会清理）
        self.target_set = set()
        # self.protected_name = ['军委联合指挥部', '中央电视台', '核工业集团机关大楼', '北京市人民防空指挥部', '空军参谋部大院']
        self.init_protected_facility = {k: v for k, v in self.side.facilities.items() if v.m_Category == 3001 and '雷达' not in v.strName}

        self.target_middle_point = (sum(t.dLatitude for t in self.init_protected_facility.values()) / len(self.init_protected_facility)
                                    , sum(t.dLongitude for t in self.init_protected_facility.values()) / len(self.init_protected_facility))
        self.weapon_dbid = {2104:'hsfw-dataweapon-00000000002104', 1225:'hsfw-dataweapon-00000000001225', 2000002:'hsfw-dataweapon-00000002000002'}

        # 防空单元类型
        self.defense_types = ['C-400', 'HQ-9A', 'HQ-12']
        self.missile_counts = [0, 1, 2, 3]  # 每种单元对单个目标的发射数量
        self.max_concurrent_targets = 24  # 最大并发处理目标数
        # self.max_concurrent_targets = 48  # 最大并发处理目标数
        # 使用MultiDiscrete空间：
        # shape = (max_targets, num_defense_types)
        # 每个元素范围 [0, 3]，表示该防空单元对该目标发射的导弹数量
        from gym.spaces import MultiDiscrete
        # 动作空间：24个目标 × 3种防空单元 = 72维（已移除do-nothing机制，始终执行拦截动作）
        # 每维取值：0(不拦截), 1(发射1枚), 2(发射2枚), 3(发射3枚)
        # 注意：已完全移除do-nothing机制，智能体必须始终根据动作空间执行拦截分配
        self.action_space = MultiDiscrete([4] * (self.max_concurrent_targets * len(self.defense_types)))
        # self.action_space = Discrete(len(self._action_func_list))
        # 动作示例：
        # action = [2, 1, 0,  # 目标1: C-400发射2枚, HQ-9A发射1枚, HQ-12不发射
        #           0, 0, 3,  # 目标2: 只有HQ-12发射3枚
        #           1, 2, 1,  # 目标3: 三种单元都参与，协同拦截
        #           0, 0, 0,  # 目标4: 不拦截
        #           ...]      # 其余目标...

        self.observation_space = Box(float("-inf"), float("inf"), shape=(20,))

        # 用于追踪当前探测到的目标列表（动态更新）
        self.current_detected_missiles = []

        # 记录每个防空单元的剩余导弹数量
        # self.missile_inventory = {
        #     'C-400': 80 * 2,  # 2个营，每营80枚
        #     'HQ-9A': 64 * 3,  # 3个营，每营64枚
        #     'HQ-12': 14 * 4  # 4个营，每营14枚
        # }

        # 记录每种类型导弹的大致总量（用于成本和“留弹意识”奖励）
        self.missile_inventory = {
            'C-400': 80 * 2,  # 2个营，每营80枚
            'HQ-9A': 64 * 3,  # 3个营，每营64枚
            'HQ-12': 14 * 4   # 4个营，每营14枚（注意重装填机制）
        }
        # 记录初始弹药总量，用于计算剩余比例
        self.initial_missile_inventory = self.missile_inventory.copy()

        # HQ-12重装填机制：初始12枚可用，打完后需2分钟重装填剩余12枚
        self.hq12_reload_status = {}  # 记录每个HQ-12单元的重装填状态
        # 记录拦截记录（用于调试，不用于奖励计算）
        self.recent_engagements = []
        # 注意：冷却机制已完全移除，不再需要engagement_cooldown
        # 单元装弹时间记录（单位：步数）
        # C-400: 80枚分为两组，每组24枚，打完一组需要重装
        # HQ-9A: 64枚分为两组，每组16枚，打完一组需要重装（约2分钟=6步）
        # HQ-12: 14枚
        self.unit_reload_time = {  # 每个单元的最后发射时间和剩余弹药
            # {unit_guid: {'last_fire_step': step, 'missiles_remaining': count}}
        }
        self.unit_reload_duration = {
            'C-400': 6,
            'HQ-9A': 6,
            'HQ-12': 6,
        }
        # 距离型奖励塑形参数（距离越近开火越好）
        self.fire_reward_near = self.env_config.get('fire_reward_near', 0.05)
        self.fire_penalty_far = self.env_config.get('fire_penalty_far', 0.05)
        self.fire_near_ratio = self.env_config.get('fire_near_ratio', 0.5)  # 距离 < 0.5 * 射程 给予正奖励
        self.fire_far_ratio = self.env_config.get('fire_far_ratio', 0.8)    # 距离 > 0.8 * 射程 给予小惩罚
        
        # ===== 新增：追踪蓝方导弹状态，用于检测拦截成功 =====
        self.detected_blue_missiles = {}  # {target_id: missile_obj} - 当前探测到的蓝方导弹
        self.intercepted_missiles = set()  # 已成功拦截的导弹ID集合
        self.missile_interception_history = {}  # {target_id: [拦截步数列表]} - 记录哪些步对该目标进行了拦截
        
        # 拦截成功奖励配置（可通过env_config配置）
        self.intercept_success_reward = self.env_config.get('intercept_success_reward', 5.0)  # 拦截成功奖励基础值
        self.intercept_failure_penalty = self.env_config.get('intercept_failure_penalty', 3.0)  # 拦截失败惩罚（目标接近但未被拦截，从1.0增大到3.0）
        self.early_intercept_bonus = self.env_config.get('early_intercept_bonus', 2.0)  # 早期拦截奖励（距离较远时拦截）
        self.early_intercept_distance = self.env_config.get('early_intercept_distance', 0.6)  # 早期拦截距离阈值（相对于射程的比例）
        # do-nothing惩罚参数
        self.do_nothing_base_penalty = self.env_config.get('do_nothing_base_penalty', 2.0)  # do-nothing基础惩罚
        self.do_nothing_target_multiplier = self.env_config.get('do_nothing_target_multiplier', 0.5)  # 每目标惩罚倍数
        self.do_nothing_c400_multiplier = self.env_config.get('do_nothing_c400_multiplier', 3.0)  # C-400可拦截目标额外惩罚倍数
        # do-nothing强制引导参数（在保留训练学习的基础上，直接降低do-nothing概率）
        self.force_action_when_targets = self.env_config.get('force_action_when_targets', True)  # 当有目标时是否强制执行动作（忽略do-nothing）
        self.force_action_target_threshold = self.env_config.get('force_action_target_threshold', 1)  # 强制执行动作的目标数量阈值（>=此数量时强制）
        self.force_action_c400_range = self.env_config.get('force_action_c400_range', True)  # 当有C-400可拦截目标时是否强制执行动作
        # 资源分配/过度开火控制 + 成本/留弹意识参数
        self.max_c400_volley = self.env_config.get('max_c400_volley', 2)  # 单目标C-400最大同发数量，避免浪费
        self.min_hq_volley_under_threat = self.env_config.get('min_hq_volley_under_threat', 2)  # 高威胁时HQ-9A/HQ-12最少发射数量
        self.under_engage_penalty = self.env_config.get('under_engage_penalty', 1)  # 高威胁目标未分配拦截的惩罚
        self.mass_threat_bonus_threshold = self.env_config.get('mass_threat_bonus_threshold', 10)  # 当空中威胁数量超过该值时鼓励多平台齐射
        self.mass_threat_bonus = self.env_config.get('mass_threat_bonus', 0.5)  # 多平台齐射的额外奖励
        # 导弹成本相关参数（用于 2.1）
        self.missile_cost_alpha = self.env_config.get('missile_cost_alpha', 0.02)  # 成本缩放系数
        # 留弹意识相关参数（用于 2.2）
        self.early_phase_threshold = self.env_config.get('early_phase_threshold', 0.5)   # 时间 < 50% 视为前期
        self.late_phase_threshold = self.env_config.get('late_phase_threshold', 0.8)    # 时间 > 80% 视为后期
        self.overuse_ratio_threshold = self.env_config.get('overuse_ratio_threshold', 0.7)  # 前期消耗>70%视为过度
        self.underuse_ratio_threshold = self.env_config.get('underuse_ratio_threshold', 0.7)  # 后期剩余>70%视为过保守
        self.early_overuse_penalty = self.env_config.get('early_overuse_penalty', 1.0)
        self.late_underuse_penalty = self.env_config.get('late_underuse_penalty', 0.5)
        # 记录最近一步各类导弹发射数量（用于成本计算）
        self.last_step_fire_stats = {'C-400': 0, 'HQ-9A': 0, 'HQ-12': 0}
        
        # ===== 基于规则的导弹发射优化参数 =====
        # C-400规则参数
        self.c400_min_distance = self.env_config.get('c400_min_distance', 100)  # 最小发射距离（km）
        self.c400_max_distance = self.env_config.get('c400_max_distance', 150)  # 最大发射距离（km）
        self.c400_reserve_ratio = self.env_config.get('c400_reserve_ratio', 0.2)  # 保留弹药比例
        self.c400_step_max = self.env_config.get('c400_step_max', 8)  # 单步最大发射数
        
        # HQ-9A规则参数
        self.hq9a_min_distance = self.env_config.get('hq9a_min_distance', 50)  # 最小发射距离（km）
        self.hq9a_max_distance = self.env_config.get('hq9a_max_distance', 100)  # 最大发射距离（km）
        self.hq9a_optimal_distance_min = self.env_config.get('hq9a_optimal_distance_min', 60)  # 最佳拦截距离下限
        self.hq9a_optimal_distance_max = self.env_config.get('hq9a_optimal_distance_max', 80)  # 最佳拦截距离上限
        self.hq9a_reserve_ratio = self.env_config.get('hq9a_reserve_ratio', 0.15)  # 保留弹药比例
        
        # 时间进度阈值
        self.early_phase = self.env_config.get('early_phase', 0.3)  # 前期阈值
        self.late_phase = self.env_config.get('late_phase', 0.7)  # 后期阈值
        
        # 是否启用基于规则的修正（默认启用）
        self.enable_rule_based_correction = self.env_config.get('enable_rule_based_correction', True)

    def _get_win_score(self):
        """
        计算当前局势的得分，用于奖励塑形（改进版）。
        
        核心思想：
            - 拦截成功奖励已在 _check_interception_success 中通过 temp_reward 添加
            - 增大保护目标损失的惩罚，使信号更明显
            - 缩小存活奖励，避免信号过密
            - 添加拦截成功率奖励
        
        Returns:
            float: 归一化的得分（除以100，方便RL使用）。
        """
        score = 0.0
        
        # 1. 计算保护目标损失惩罚（动态惩罚：损失越多惩罚越大）
        self.protected_target = {k: v.strName for k, v in self.side.facilities.items()
                                 if v.m_Category == 3001 and '雷达' not in v.strName}
        lost_count = len(self.init_protected_facility) - len(self.protected_target)
        # 使用递增惩罚：第1个损失50，第2个损失60，第3个损失70...
        # 这样可以更强烈地惩罚连续损失
        base_penalty = 50.0
        penalty = sum(base_penalty + i * 10.0 for i in range(lost_count))
        score -= penalty
        
        # 2. 添加保护目标存活的奖励（合并存活奖励逻辑）
        detected_missiles = {k: v for k, v in self.side.contacts.items() if v.m_ContactType == 1}
        has_threat = len(detected_missiles) > 0
        
        if has_threat and len(self.protected_target) > 0:
            remaining_count = len(self.protected_target)
            # 存活目标奖励：每存活一个目标，每步给0.3分奖励（鼓励保护）
            score += remaining_count * 0.5
        
        # 3. 若所有保护目标被摧毁，叠加一次性终局大惩罚
        if len(self.protected_target) == 0 and lost_count > 0:
            score -= 500.0  # 从200.0增大到500.0
        
        # 4. 拦截统计奖励（新增：鼓励拦截更多目标）
        total_encountered = len(detected_missiles) + len(self.intercepted_missiles)
        if total_encountered > 0:
            intercept_rate = len(self.intercepted_missiles) / total_encountered
            # 拦截率奖励：拦截率越高奖励越大，最高5.0分（从2.0增大到5.0）
            score += intercept_rate * 5.0

        # 5. 多波次场景下的“留弹意识”（2.2）
        # 使用时间进度 + 剩余弹药比例进行软约束
        time_delta = self.m_Time - self.m_StartTime
        # 想定总时长可能为 self.time[-1]，但这里用经验值做归一（与观测空间一致）
        # 以 5400 秒（1.5 小时）作为最大尺度
        time_frac = max(0.0, min(1.0, time_delta / 5400.0))

        # 计算三类导弹的剩余比例
        remaining_ratios = {}
        for k, v in self.missile_inventory.items():
            init_v = self.initial_missile_inventory.get(k, v)
            remaining_ratios[k] = v / init_v if init_v > 0 else 1.0

        # 前期过度消耗惩罚：时间还早但某类导弹已消耗过多
        if time_frac < self.early_phase_threshold:
            for k, ratio in remaining_ratios.items():
                if (1.0 - ratio) > self.overuse_ratio_threshold:
                    score -= self.early_overuse_penalty

        # 后期过于保守惩罚：时间接近结束但某类导弹仍大量剩余且拦截率不高
        if time_frac > self.late_phase_threshold and total_encountered > 0 and intercept_rate < 0.8:
            for k, ratio in remaining_ratios.items():
                if ratio > self.underuse_ratio_threshold:
                    score -= self.late_underuse_penalty

        # ===== 改进：前100步不输出 =====
        if self.steps > 100 and self.steps % 10 == 0:
            print(f'红方地面核心设施还剩下{len(self.protected_target)-1}个，'
                  f'已拦截{len(self.intercepted_missiles)}个目标，得分{score/100:.2f}')
        return float(score) / 100

    def _check_interception_success(self):
        """
        检测拦截成功的导弹
        
        原理：
        1. 对比上一 step 和当前 step 探测到的导弹列表
        2. 如果某个导弹消失了，且最近几步对它进行过拦截，则认为是拦截成功
        3. 给予拦截成功奖励
        
        Returns:
            int: 本次检测到的拦截成功数量
        """
        # 获取当前探测到的所有蓝方导弹
        current_detected = {k: v for k, v in self.side.contacts.items() 
                           if v.m_ContactType == 1}  # m_ContactType == 1 表示导弹
        
        current_ids = set(current_detected.keys())
        previous_ids = set(self.detected_blue_missiles.keys())
        
        # 找出消失的导弹（可能被拦截了）
        disappeared_ids = previous_ids - current_ids
        
        interception_count = 0
        for missile_id in disappeared_ids:
            # 检查该导弹是否在最近几步被拦截过
            if missile_id in self.missile_interception_history:
                # 检查最近几步（例如最近10步）是否进行过拦截
                recent_engagement_steps = [
                    step for step in self.missile_interception_history[missile_id]
                    if (self.steps - step) <= 10
                ]
                
                if recent_engagement_steps:
                    # 认为是拦截成功
                    if missile_id not in self.intercepted_missiles:
                        self.intercepted_missiles.add(missile_id)
                        
                        # 根据拦截时的距离给予奖励（早期拦截给更多奖励）
                        # 尝试从拦截历史中找到拦截时的距离信息
                        intercept_step = recent_engagement_steps[0]
                        reward = self.intercept_success_reward
                        
                        # 如果能找到拦截时的距离信息，给予早期拦截奖励
                        # 这里简化处理，如果有早期拦截历史记录，额外奖励
                        if len(recent_engagement_steps) > 0:
                            # 查找最早的拦截步数，如果是较早拦截，给予奖励
                            earliest_step = min(recent_engagement_steps)
                            # 如果是在较远的距离拦截（步数较早），给予额外奖励
                            # 简化：如果拦截步数早于当前步数5步以上，认为是早期拦截
                            if (self.steps - earliest_step) >= 5:
                                reward += self.early_intercept_bonus
                        
                        self.temp_reward += reward
                        interception_count += 1
                        # ===== 改进：前100步不输出拦截成功信息 =====
                        if self.steps > 100:
                            if reward > self.intercept_success_reward:
                                print(f"[步骤{self.steps}] 拦截成功（早期拦截）！导弹 {missile_id} 被击落，奖励 +{reward:.2f}")
                            else:
                                print(f"[步骤{self.steps}] 拦截成功！导弹 {missile_id} 被击落，奖励 +{reward:.2f}")
            
            # 清理该导弹的历史记录
            if missile_id in self.missile_interception_history:
                del self.missile_interception_history[missile_id]
        
        # 更新当前探测到的导弹列表
        self.detected_blue_missiles = current_detected
        
        return interception_count

    def _check_interception_failure(self):
        """
        检测拦截失败（目标接近保护目标但未被拦截）
        
        对于距离保护目标很近但仍未被拦截的导弹，给予惩罚
        距离越近惩罚越大，鼓励尽早拦截，避免等到最后一刻
        """
        failure_count = 0
        total_penalty = 0.0
        
        for missile_id, missile in self.detected_blue_missiles.items():
            if missile_id in self.intercepted_missiles:
                continue  # 已拦截，跳过
            
            distance = self._get_distance_to_protected_targets(missile)
            # 如果距离很近（<30km）且没有被拦截，说明拦截可能失败
            # 距离越近，惩罚越大（鼓励早期拦截）
            if distance < 30:
                failure_count += 1
                # 距离越近惩罚越大：30km时1倍，20km时1.5倍，10km时2倍
                penalty_multiplier = 1.0 + (30 - distance) / 20.0
                penalty = self.intercept_failure_penalty * penalty_multiplier
                total_penalty += penalty
        
        if total_penalty > 0:
            self.temp_reward -= total_penalty
            # ===== 改进：前100步不输出 =====
            if self.steps > 100 and self.steps % 10 == 0:
                print(f"[步骤{self.steps}] 检测到 {failure_count} 个目标接近保护区域且未被拦截，惩罚 -{total_penalty:.2f}")
        
        return failure_count

    def _update(self, scenario):
        """
        用最新的想定态势更新内部引用，并刷新奖励基线。

        Args:
            scenario: 墨子返回的最新想定对象。
        """
        self.side = scenario.get_side_by_name(self.side_name)
        
        # ===== 新增：检测C-400单元是否被摧毁 =====
        current_c400_count = len([v for k, v in self.side.facilities.items() if 'C-400' in v.strName])
        if hasattr(self, 'previous_c400_count') and current_c400_count < self.previous_c400_count:
            # C-400单元被摧毁，给予大惩罚（鼓励早期拦截保护C-400）
            lost_c400 = self.previous_c400_count - current_c400_count
            penalty = -50.0 * lost_c400  # 每个被摧毁的C-400单元惩罚50分
            self.temp_reward += penalty
            if self.steps > 100:
                print(f'[步骤{self.steps}] 严重警告：{lost_c400}个C-400单元被摧毁！惩罚-{abs(penalty):.2f}（提示：应该尽早拦截保护C-400）')
        # 更新C-400计数（无论是否被摧毁都要更新）
        self.previous_c400_count = current_c400_count
        # ===== 新增代码结束 =====
        
        current_score = self._get_win_score()
        self.reward = current_score - self.reward_accum + self.temp_reward
        self.reward_accum = current_score + self.temp_reward
        # 临时奖励在消费后清零
        self.temp_reward = 0
        self.m_Time = self.scenario.m_Time  # 想定当前时间

    def step(self, action_dict):
        """
        RL标准接口：接收动作字典，执行拦截计划并返回新状态/奖励。

        Args:
            action_dict: 包含唯一智能体`agent_0`的动作向量。

        Returns:
            tuple: (obs, reward, done_info, extra_info)
        """
        done = False
        action = action_dict['agent_0']
        # 1. 解析动作
        engagement_plan = self._parse_action(action)

        if self.env_config['mode'] in ['train', 'development']:
            force_done = self.safe_step(action, engagement_plan)
            if force_done:
                done = force_done
                self.reset_nums = 4  # 下一局会重启墨子docker(每5局重启一次docker)
                print(f"{time.strftime('%H:%M:%S')} 在第{self.steps}步，强制重启墨子！！！")
            else:
                # ===== 新增：在更新态势前检测拦截成功 =====
                # 需要先临时更新side以获取最新态势来检测拦截
                if self.scenario:
                    old_side = self.side
                    self.side = self.scenario.get_side_by_name(self.side_name)
                    interception_count = self._check_interception_success()
                    failure_count = self._check_interception_failure()
                    self.side = old_side  # 恢复，让_update正常更新
                # ===== 新增代码结束 =====
                self._update(self.scenario)
                done = self._is_done()
        elif self.env_config['mode'] in ['versus', 'eval']:
            self._execute_engagement_plan(engagement_plan)
            self.scenario = self.env.step()  # 墨子环境step
            # ===== 新增：检测拦截成功 =====
            # 需要先临时更新side以获取最新态势来检测拦截
            if self.scenario:
                old_side = self.side
                self.side = self.scenario.get_side_by_name(self.side_name)
                interception_count = self._check_interception_success()
                failure_count = self._check_interception_failure()
                self.side = old_side  # 恢复，让_update正常更新
            # ===== 新增代码结束 =====
            self._update(self.scenario)
            done = self._is_done()
        reward = {'agent_0': self.reward}
        obs_array = np.array(self._generate_features(), dtype=np.float32)
        obs = {'agent_0': obs_array}
        # if isinstance(obs, dict) and 'obs' in obs:
        #     obs = np.array(obs['obs'], dtype=np.float32)
        # elif isinstance(obs, list):
        #     obs = np.array(obs, dtype=np.float32)
        self.steps += 1
        # ===== 改进：前100步不输出详细信息（减少日志噪音） =====
        if self.steps > 100:
            if self.steps % 10 == 0:
                print(f'第 {self.steps}步' + '-' + f'reward is {self.reward}')
        if done:
            print('++++Score:', self.reward_accum, 'step:', self.steps)
            print(f'本局拦截成功数量: {len(self.intercepted_missiles)}')
            # 计算导弹消耗（初始数量 - 当前剩余数量）
            missile_consumption = {
                'C-400': self.initial_missile_inventory.get('C-400', 0) - self.missile_inventory.get('C-400', 0),
                'HQ-9A': self.initial_missile_inventory.get('HQ-9A', 0) - self.missile_inventory.get('HQ-9A', 0),
                'HQ-12': self.initial_missile_inventory.get('HQ-12', 0) - self.missile_inventory.get('HQ-12', 0),
            }
            # 计算导弹消耗总费用
            total_missile_cost = (
                RED_MISSILE_COST.get('C-400', 0) * missile_consumption['C-400'] +
                RED_MISSILE_COST.get('HQ-9A', 0) * missile_consumption['HQ-9A'] +
                RED_MISSILE_COST.get('HQ-12', 0) * missile_consumption['HQ-12']
            )
            print(f'本局导弹消耗: C-400={missile_consumption["C-400"]}, '
                  f'HQ-9A={missile_consumption["HQ-9A"]}, HQ-12={missile_consumption["HQ-12"]}, '
                  f'总费用={total_missile_cost:.1f}')
        else:
            missile_consumption = {'C-400': 0, 'HQ-9A': 0, 'HQ-12': 0}
            total_missile_cost = 0.0
        
        # ===== 返回额外信息用于监控 =====
        extra_info = {
            'agent_0': {
                'score': self._get_win_score(),
                'protected_facilities': len(self.protected_target),
                'intercepted_missiles': len(self.intercepted_missiles),
                'episode_reward': self.reward_accum,
                'episode_steps': self.steps,
                'missile_consumption': missile_consumption,  # 导弹消耗数量
                'missile_cost': total_missile_cost,  # 导弹消耗总费用
            }
        }
        return obs, reward, {'__all__': done, 'agent_0': done}, extra_info

    def safe_step(self, action, engagement_plan):
        """
        在训练/开发模式下的安全step包装，避免lua执行或态势更新异常导致崩溃。

        Args:
            action: 原始动作（未使用，仅兼容接口）。
            engagement_plan: 已解析的拦截计划。

        Returns:
            bool: True表示需强制结束本局（例如lua超时或态势更新失败）。
        """
        force_done = False
        # noinspection PyBroadException
        try:
            self._execute_engagement_plan(engagement_plan)
        except Exception:
            print(f"{time.strftime('%H:%M:%S')} 在第{self.steps}步，执行lua超时！！！")
            force_done = True
            return force_done
        # noinspection PyBroadException
        try:
            self.scenario = self.env.step()  # 墨子环境step
        except Exception:
            print(f"{time.strftime('%H:%M:%S')} 在第{self.steps}步，更新态势超时！！！")
            force_done = True
            return force_done
        if self.scenario and self.scenario.get_side_by_name(self.side_name):
            return force_done
        else:
            # 态势更新失败会抛出异常
            print(f"{time.strftime('%H:%M:%S')} 在第{self.steps}步，更新态势失败！！！")
            force_done = True
            return force_done

    def reset(self):
        """
        RL标准接口：重置环境并返回初始观测。

        重启逻辑：
            - 每5局在训练/开发模式下重启docker，防止状态漂移。
            - 清空奖励与拦截记录，重新计算基线得分。
        """
        self._get_initial_state()
        self.steps = 0
        self.temp_reward = 0
        # 清理上一局的状态，确保奖励在新一局开始时不带有历史数据
        self.recent_engagements = []
        # ===== 新增：清理拦截追踪变量 =====
        self.detected_blue_missiles = {}
        self.intercepted_missiles = set()
        self.missile_interception_history = {}
        self.unit_reload_time = {}  # 清理装弹时间记录
        # ===== 新增代码结束 =====
        # 在初始态势下获取基线分数，作为累积基线
        self.reward_accum = self._get_win_score()
        # 在基线之上执行一次更新以同步内部状态，保证下一步的奖励计算正确
        self._update(self.scenario)
        # 初始化本局即时奖励为0
        self.reward = 0.0

        # 限制所有防空单元不开火（对所有C-400、HQ-9A、HQ-12单元都设置）
        red_facilities = self.side.get_facilities()  # 使用正确的方法
        # print(f"红方所有单元{red_facilities}")
        for unit_id, unit in red_facilities.items():
            unit_name = getattr(unit, 'strName', '')
            
            # 只对防空单元（C-400、HQ-9A、HQ-12）设置限制开火状态
            is_defense_unit = False
            unit_type = None
            
            if 'C-400' in unit_name:
                is_defense_unit = True
                unit_type = 'C-400'
            elif 'HQ-9A' in unit_name:
                is_defense_unit = True
                unit_type = 'HQ-9A'
            # elif '红旗-12' in unit_name or 'HQ-12' in unit_name:
            #     is_defense_unit = True
            #     unit_type = 'HQ-12'
            
            if is_defense_unit:
                # 设置单元条令的对空状态为限制开火
                unit_doctrine = unit.get_doctrine()
                #参数：fire_status: {str: '0' - 自由开火, '1' - 谨慎开火, '2' - 限制开火}
                unit_doctrine.set_weapon_control_status_air(2)
                print(f"已设置{unit_type}单元 {unit_name} 为限制开火状态")



        # ===== 新增：初始化探测到的导弹列表 =====
        self.detected_blue_missiles = {k: v for k, v in self.side.contacts.items() 
                                       if v.m_ContactType == 1}
        # ===== 新增：记录C-400单元状态，用于检测是否被摧毁 =====
        self.previous_c400_count = len([v for k, v in self.side.facilities.items() if 'C-400' in v.strName])
        # ===== 新增代码结束 =====
        obs = {'agent_0': np.array(self._generate_features(), dtype=np.float32)}
        print('env_reset finished!!!')
        return obs

    def _generate_features(self):
        """
        构造观测特征向量（20维），覆盖蓝方导弹态势、红方防空单元状态与时间进度。

        Returns:
            list[float]: 归一化后的特征列表，后续会转为np.array。
        """
        feats = []  # 总共20维度

        # 1. 蓝方来袭导弹信息 (7维)
        blue_missiles = {k: v for k, v in self.side.contacts.items()
                         if v.m_ContactType == 1}  # 仅保留探测到的导弹型目标

        blue_missile_feats = [0.0] * 7
        div = 0.0
        for k, v in blue_missiles.items():
            div += 1.0  # 统计导弹数量用于均值化
            temp_feats = [0.0] * 12  # 临时特征容器，超过7维但最后取均值
            temp_feats[0] = v.m_ContactType / 22.0  # 目标类型归一化
            # 位置信息
            temp_feats[1] = v.dLongitude / 180.0  # 经度归一化
            temp_feats[2] = v.dLatitude / 180.0  # 纬度归一化

            # 速度和航向
            temp_feats[3] = v.fCurrentSpeed / 1000.0  # 速度缩放
            temp_feats[4] = v.fCurrentHeading / 360.0  # 航向角归一化

            # 识别状态
            temp_feats[5] = v.m_IdentificationStatus / 4.0  # 识别等级

            # 高度信息，探测到的导弹一般是低空突防
            temp_feats[6] = v.fCurrentAltitude_ASL / 1000.0 if hasattr(v, 'fCurrentAltitude_ASL') else 0.0  # 高度可选项

            blue_missile_feats = list(map(lambda x, y: x + y, blue_missile_feats, temp_feats))

        if div > 0:
            blue_missile_feats = [f / div for f in blue_missile_feats]  # 对多个导弹取平均特征
        feats.extend(blue_missile_feats)

        # 2. 红方防空单元状态 (9维)
        # C-400状态
        c400_units = [v for k, v in self.side.facilities.items() if 'C-400' in v.strName]  # 筛选所有C-400营
        c400_feats = self._get_unit_status_feats(c400_units, 'C-400')
        feats.extend(c400_feats)

        # HQ-9A状态
        hq9_units = [v for k, v in self.side.facilities.items() if 'HQ-9A' in v.strName]  # 筛选HQ-9A营
        hq9_feats = self._get_unit_status_feats(hq9_units, 'HQ-9A')
        feats.extend(hq9_feats)

        # HQ-12状态
        hq12_units = [v for k, v in self.side.facilities.items() if '红旗-12' in v.strName or 'HQ-12' in v.strName]  # 筛选HQ-12营
        hq12_feats = self._get_unit_status_feats(hq12_units, 'HQ-12')
        feats.extend(hq12_feats)

        time_delta = self.m_Time - self.m_StartTime   # 想定基本1个半小时结束
        feats.append(time_delta / 1800.0)  # 时间进度特征1
        feats.append(time_delta / 3600.0)  # 时间进度特征2
        feats.append(time_delta / 4500.0)  # 时间进度特征3
        feats.append(time_delta / 5400.0)  # 时间进度特征4
        return feats

    def _get_unit_status_feats(self, units, unit_type):
        """获取单一类型防空单元的状态特征"""
        if not units:
            return [0.0] * 3  # 没有该类型单元时返回零向量

        total_missiles = 0  # 累计导弹数
        avg_position = [0.0, 0.0]  # 累计经纬度用于求均值
        avg_readiness = 0.0  # 预留 readiness 指标（当前未使用）

        for unit in units:
            # 获取可用导弹数量
            weapon_list = self._get_unit_weapon(unit)
            missile_count = self._get_weapon_num(weapon_list, [2000002, 1225, 2104])
            total_missiles += missile_count

            avg_position[0] += unit.dLongitude  # 经度累加
            avg_position[1] += unit.dLatitude  # 纬度累加

        num_units = len(units)
        return [
            total_missiles / (num_units * RED_MISSILE_AVAILABLE[unit_type]),  # 剩余导弹比例
            avg_position[0] / (num_units * 180.0),  # 平均经度
            avg_position[1] / (num_units * 180.0),  # 平均纬度
        ]

    @staticmethod
    def _get_unit_weapon(unit):
        """
        :param unit: aircraft, ship
        :return:
        """
        weapon = list(map(lambda x: x.split('$'), unit.m_UnitWeapons.split('@')))  # 先按@分组再按$拆分字段
        weapon_list = list(map(lambda x, y: x + [y[-1]], list(map(lambda x: x[0].split('x '), weapon)), weapon))  # 拼出[数量,名称,dbid]
        return weapon_list  # 结构示例：[['4', 'HQ-9A', 'hsfw-dataweapon-...']]

    @staticmethod
    def _get_weapon_num(weapon_list, weapon_type):
        num = 0  # 计数器
        for weapon in weapon_list:
            if weapon[0] != '' and weapon[-1] != '':
                if int(re.sub('\D', '', weapon[-1])) in weapon_type:
                    num += int(weapon[0])
        return num

    def _get_env(self):
        """
        根据模式(train/development/versus/eval)创建并启动墨子环境。

        Notes:
            - train/development/versus使用同一linux想定。
            - eval模式使用windows平台与评测想定。
        """
        if self.env_config['mode'] == 'train':
            self.schedule_addr = self.env_config['schedule_addr']
            self.schedule_port = self.env_config['schedule_port']
            scenario_name = etc.SCENARIO_NAME
            platform = 'linux'
            self._create_env(platform, scenario_name=scenario_name)
        elif self.env_config['mode'] == 'development':
            scenario_name = etc.SCENARIO_NAME
            platform = 'linux'
            self._create_env(platform, scenario_name=scenario_name)
        elif self.env_config['mode'] == 'versus':
            scenario_name = etc.SCENARIO_NAME
            platform = 'linux'
            self._create_env(platform, scenario_name=scenario_name)
        elif self.env_config['mode'] == 'eval':
            scenario_name = etc.EVAL_SCENARIO_NAME
            platform = 'windows'
            self._create_env(platform, scenario_name=scenario_name)

            # platform = 'linux'
            # self._create_env(platform)
        else:
            raise NotImplementedError

    def _create_env(self, platform, scenario_name=None):
        """
        尝试多次启动墨子环境，并绑定可用docker端口。

        Args:
            platform: 'linux' 或 'windows'，决定仿真运行平台。
            scenario_name: 想定名称，若None则使用默认。
        """
        for _ in range(MAX_DOCKER_RETRIES):
            # noinspection PyBroadException
            try:
                self.env = Environment(etc.SERVER_IP,
                                       etc.SERVER_PORT,
                                       platform,
                                       scenario_name,
                                       etc.SIMULATE_COMPRESSION,
                                       etc.DURATION_INTERVAL,
                                       etc.SYNCHRONOUS)
                # by dixit
                if self.env_config['avail_docker_ip_port']:
                    self.avail_ip_port_list = self.env_config['avail_docker_ip_port']
                else:
                    raise Exception('no avail port!')
                # self.self.reset_nums = 0
                self.ip_port = self.avail_ip_port_list[0]  # 默认取第一个可用docker
                print(self.ip_port)
                self.ip = self.avail_ip_port_list[0].split(":")[0]
                self.port = self.avail_ip_port_list[0].split(":")[1]
                self.ip_port = f'{self.ip}:{self.port}'
                self.env.start(self.ip, self.port)  # 启动仿真服务
                break
            except Exception:
                continue

    def _get_initial_state(self):
        """
        dixit 2021/3/22
        每5局重启墨子，获取初始态势
        """
        # 记录reset次数，基于此触发周期性docker重启

        self.reset_nums += 1
        if self.env_config['mode'] in ['train', 'development']:
            if self.reset_nums % 5 == 0:
                docker_ip_port = self.avail_ip_port_list[0]
                for _ in range(MAX_DOCKER_RETRIES):
                    # noinspection PyBroadException
                    try:
                        if self.env_config['mode'] == 'train':
                            restart_container(self.schedule_addr,
                                              self.schedule_port,
                                              self.env_config['training_id'],
                                              docker_ip_port)
                        else:
                            restart_mozi_container(docker_ip_port)
                        self.env = Environment(etc.SERVER_IP,
                                               etc.SERVER_PORT,
                                               'linux',
                                               etc.SCENARIO_NAME,
                                               etc.SIMULATE_COMPRESSION,
                                               etc.DURATION_INTERVAL,
                                               etc.SYNCHRONOUS)
                        self.env.start(self.ip, self.port)  # docker重启后需重新启动仿真
                        break
                    except Exception:
                        print(f"{time.strftime('%H:%M:%S')} 在第{self.steps}步，第{_}次重启docker失败！！！")
                        continue
                print('开始mozi reset!!!')
                self.scenario = self.env.reset(self.side_name)
                print('结束mozi reset!!!')
            else:
                print('开始mozi reset!!!')
                self.scenario = self.env.reset(self.side_name)
                print('结束mozi reset!!!')
        else:
            self.scenario = self.env.reset(self.side_name)

    def _is_done(self):
        """
        结束判定：
        1）仿真平台返回 EndOfDeduction
        2）所有红方地面核心设施被摧毁（self.protected_target 为空）
        """
        # 1. 平台内部结束标记
        response_dic = self.scenario.get_responses()
        for _, v in response_dic.items():
            if v.Type == 'EndOfDeduction':
                print('打印出标记：EndOfDeduction')
                return True

        # 2. 红方保护目标全部被摧毁
        # 与 _get_win_score 中的统计方式保持一致：m_Category == 3001
        current_protected = {k: v for k, v in self.side.facilities.items() if getattr(v, 'm_Category', None) == 3001 and '雷达' not in v.strName}
        if len(current_protected) == 0:
            print('所有红方地面核心设施被摧毁，本局结束')
            return True

        return False

    # 《《《《《《《《《《《《《 动作空间 》》》》》》》》》》》》》》

    def _parse_action(self, action):
        """
        解析动作向量为具体的拦截指令

        Args:
            action: MultiDiscrete动作，shape=(72,)
                    前24个元素是所有目标对C-400的指令
                    中间24个是对HQ-9A的指令
                    后24个是对HQ-12的指令
                    注意：已完全移除do-nothing机制，动作空间现在是72维

        Returns:
            engagement_plan: 字典列表，每个元素代表一次拦截指令
            [
                {
                    'target_id': 目标ID,
                    'target': 目标对象,
                    'defense_assignments': {
                        'C-400': 2,  # 发射2枚
                        'HQ-9A': 1,  # 发射1枚
                        'HQ-12': 0   # 不发射
                    }
                },
                ...
            ]
        """
        # 获取当前探测到的所有蓝方导弹（不再统一排序，因为威胁度是基于每个平台类型计算的）
        detected_missiles = self._get_detected_missiles_sorted()  # 返回所有探测到的目标列表
        
        # 注意：已完全移除do-nothing机制，动作空间现在是72维（24目标 × 3种防空单元）
        # 直接使用完整的动作向量，不需要判断do-nothing标志
        
        # 将动作重塑为 (24, 3) 的矩阵
        # 每行代表一个目标，每列代表一种防空单元的发射数量
        action_matrix = action.reshape(self.max_concurrent_targets, len(self.defense_types))  # reshape便于逐目标解析

        engagement_plan = []

        # 只处理实际探测到的目标数量（最多24个）
        num_detected = min(len(detected_missiles), self.max_concurrent_targets)
        
        # ===== 改进：基于每个平台类型的威胁度来构建拦截计划 =====
        # 对每个目标，计算它对每个平台类型的威胁度，并基于威胁度来调整动作分配
        
        for i in range(num_detected):
            target_id, target = detected_missiles[i]

            # 获取智能体原始动作分配
            c400_count = int(action_matrix[i, 0])  # C-400发射数量
            hq9a_count = int(action_matrix[i, 1])  # HQ-9A发射数量
            hq12_count = int(action_matrix[i, 2])  # HQ-12发射数量

            # 计算该目标对每个平台类型的威胁度
            c400_threat = self._calculate_threat_score_for_platform(target, 'C-400')
            hq9a_threat = self._calculate_threat_score_for_platform(target, 'HQ-9A')
            hq12_threat = self._calculate_threat_score_for_platform(target, 'HQ-12')

            # 获取距离信息（用于后续的距离检查）
            c400_distance = self._get_distance_to_platform_type(target, 'C-400')
            hq9a_distance = self._get_distance_to_platform_type(target, 'HQ-9A')
            hq12_distance = self._get_distance_to_platform_type(target, 'HQ-12')

            # 基于威胁度来调整动作分配（如果威胁度很高但动作分配为0，可以强制分配）
            # 注意：这里我们主要记录威胁度信息，实际的拦截分配仍然由智能体决定
            # 但可以在奖励机制中根据威胁度来给予更强的信号
            
            # 构建该目标的防御分配
            defense_assignments = {
                'C-400': c400_count,
                'HQ-9A': hq9a_count,
                'HQ-12': hq12_count
            }

            # 计算目标到保护目标的距离（保留用于兼容性）
            target_distance = self._get_distance_to_protected_targets(target)

            # 记录至少有一种单元参与拦截的目标，或虽然未分配但威胁度较高的目标（用于奖励机制）
            if sum(defense_assignments.values()) > 0:
                engagement_plan.append({
                    'target_id': target_id,
                    'target': target,
                    'defense_assignments': defense_assignments,
                    'target_distance': target_distance,  # 保留用于兼容性
                    'target_threat': max(c400_threat, hq9a_threat, hq12_threat),  # 使用最高威胁度
                    # 新增：针对每个平台类型的威胁度和距离信息
                    'platform_threats': {
                        'C-400': c400_threat,
                        'HQ-9A': hq9a_threat,
                        'HQ-12': hq12_threat
                    },
                    'platform_distances': {
                        'C-400': c400_distance,
                        'HQ-9A': hq9a_distance,
                        'HQ-12': hq12_distance
                    }
                })

        # ===== 调试输出：显示威胁度统计信息 =====
        if len(detected_missiles) > 0 and self.steps > 0 and self.steps % 5 == 0:
            c400_range = RED_MISSILE_RANGE['C-400']
            c400_detectable_count = 0
            c400_high_threat_count = 0
            c400_assigned_count = 0
            for plan in engagement_plan:
                c400_dist = plan.get('platform_distances', {}).get('C-400', 999.0)
                c400_threat = plan.get('platform_threats', {}).get('C-400', 0.0)
                if c400_dist <= c400_range:
                    c400_detectable_count += 1
                if c400_threat >= 8.0:  # 高威胁阈值
                    c400_high_threat_count += 1
                if plan['defense_assignments'].get('C-400', 0) > 0:
                    c400_assigned_count += 1
            if c400_detectable_count > 0:
                print(f'[步骤{self.steps}] 探测到{len(detected_missiles)}个蓝方导弹，其中{c400_detectable_count}个在C-400射程内（≤{c400_range}km），{c400_high_threat_count}个对C-400威胁度高（≥8.0），已分配{c400_assigned_count}个C-400拦截')
        # ===== 调试输出结束 =====

        return engagement_plan

    def _execute_engagement_plan(self, engagement_plan):
        """
        执行拦截计划

        核心逻辑：
        1. 应用基于规则的修正（如果启用）
        2. 检查导弹库存是否充足
        3. 选择最优的单元执行拦截（考虑位置、射程、导弹数量）
        4. 调用手动打击接口
        5. 更新导弹库存
        6. 处理HQ-12重装填机制

        Args:
            engagement_plan: _parse_action返回的拦截计划
        """
        # ===== 应用基于规则的修正 =====
        if self.enable_rule_based_correction:
            engagement_plan, rule_step_fire_stats = self._apply_rule_based_correction(engagement_plan)
        else:
            rule_step_fire_stats = {'C-400': 0, 'HQ-9A': 0, 'HQ-12': 0}
        
        executed_engagements = []
        # 本步各类导弹发射计数（用于成本奖励 2.1）
        step_fire_stats = {'C-400': 0, 'HQ-9A': 0, 'HQ-12': 0}

        for plan in engagement_plan:
            target_id = plan['target_id']  # 目标唯一ID
            target = plan['target']  # 目标对象
            assignments = plan['defense_assignments']  # 三类防空分配
            target_distance = plan['target_distance']  # 目标距离保护中心

            # ===== 新增：记录拦截历史 =====
            if target_id not in self.missile_interception_history:
                self.missile_interception_history[target_id] = []
            
            # 如果有任何拦截动作，记录本次拦截
            if any(count > 0 for count in assignments.values()):
                self.missile_interception_history[target_id].append(self.steps)
            # ===== 新增代码结束 =====

            # 注意：冷却机制已完全移除，不再检查engagement_cooldown

            # 对每种防空单元执行拦截
            for defense_type, missile_count in assignments.items():
                if missile_count == 0:
                    continue

                # 改进：基于平台特定的威胁度和距离来给予奖励/惩罚
                platform_threats = plan.get('platform_threats', {})
                platform_distances = plan.get('platform_distances', {})
                
                # 获取该平台类型的威胁度和距离
                platform_threat = platform_threats.get(defense_type, 0.0)
                platform_distance = platform_distances.get(defense_type, 999.0)
                max_range = RED_MISSILE_RANGE[defense_type]
                
                # 如果该目标对该平台威胁度高，但未分配该平台拦截，给予惩罚
                if defense_type == 'C-400':
                    c400_range = RED_MISSILE_RANGE['C-400']
                    c400_threat = platform_threat
                    c400_distance = platform_distance
                    
                    # 如果目标在C-400射程内且威胁度高，但未分配C-400拦截
                    if c400_distance <= c400_range and c400_threat >= 8.0:
                        if assignments.get('C-400', 0) == 0 and sum(assignments.values()) == 0:
                            # 完全没有拦截分配，给予较大惩罚
                            threat_factor = c400_threat / 10.0  # 威胁度越高，惩罚越大
                            distance_factor = 1.0 + (c400_distance - 70) / 110.0  # 70km时1.0，180km时2.0
                            penalty = self.under_engage_penalty * 3.0 * threat_factor * distance_factor
                            self.temp_reward -= penalty
                        elif assignments.get('C-400', 0) == 0 and sum(assignments.values()) > 0:
                            # 使用了其他武器但未使用C-400
                            if c400_distance >= 100:  # 远距离应该优先用C-400
                                penalty = self.under_engage_penalty * 1.5 * (c400_threat / 10.0)
                                self.temp_reward -= penalty
                
                # 如果目标对该平台威胁度高但未分配足够弹药，给予小惩罚
                if platform_threat >= 8.0 and missile_count == 0:
                    self.temp_reward -= self.under_engage_penalty * 0.5

                # 1. 开火数量控制/防过度浪费（如果规则修正已应用，这里只做二次检查）
                if defense_type == 'C-400' and missile_count > self.max_c400_volley:
                    missile_count = self.max_c400_volley
                # 低空近程时，鼓励HQ-9A/HQ-12至少打2发（基于平台特定的距离）
                platform_distances = plan.get('platform_distances', {})
                platform_distance = platform_distances.get(defense_type, 999.0)
                if defense_type in ['HQ-9A', 'HQ-12'] and missile_count == 0 and platform_distance < 80:
                    # 如果规则修正未启用或未修正，这里强制至少发射
                    if not self.enable_rule_based_correction:
                        missile_count = self.min_hq_volley_under_threat

                # 2. 检查射程约束（基于平台特定的距离）
                platform_distances = plan.get('platform_distances', {})
                platform_distance = platform_distances.get(defense_type, 999.0)
                max_range = RED_MISSILE_RANGE[defense_type]
                
                # 使用平台特定的距离来判断射程
                if platform_distance > max_range:
                    # 目标超出该平台类型的射程，跳过
                    continue

                # 3. 选择最优单元（距离最近、状态最好、导弹充足，且不在装弹状态
                # 说明：是从同类型平台中选择最好的那一个，比如有三个”HQ-12“，选择最优的发射）
                best_unit_dict = self._select_best_unit_for_engagement(
                    defense_type,
                    target,
                    missile_count
                )

                if best_unit_dict is None and defense_type == 'HQ-12':
                    print("------平台类型为HQ-12，但是返回为空-------")
                    print("---------------------------------------------------------")
                    # 在新线程中显示警告窗口，避免阻塞主程序
                    # warning_thread = threading.Thread(target=show_warning)
                    # warning_thread.start()
                    # 在控制台显示醒目的警告
                    print("\n" + "⚠️" * 30)
                    print("⚠️ 警告：平台类型为HQ-12，但是返回为空！")
                    print("⚠️" * 30 + "\n")
                    # Windows系统通知
                    if platform.system() == "Windows":
                        try:
                            ctypes.windll.user32.MessageBoxW(0,
                                                             "平台类型为HQ-12，但是返回为空！\n请检查相关配置。",
                                                             "HQ-12警告",
                                                             0x30)  # MB_ICONWARNING
                        except:
                            pass

                if not best_unit_dict:
                    continue
                
                # ===== 检查单元装弹状态（改进版） =====
                # 注意：装弹时间检查应该在选择单元时考虑，这里我们简化处理
                # 实际的装弹机制由墨子平台内部管理，我们这里主要记录发射历史
                # 如果需要更精确的装弹控制，需要从墨子平台获取单元的实时装弹状态

                # 临时奖励（改进：基于平台特定的威胁度和距离来给予奖励）
                platform_threats = plan.get('platform_threats', {})
                platform_distances = plan.get('platform_distances', {})
                platform_threat = platform_threats.get(defense_type, 0.0)
                platform_distance = platform_distances.get(defense_type, 999.0)
                
                if defense_type == 'C-400' and missile_count >= 1:
                    # C-400发射基础奖励
                    base_reward = 0.1
                    # 基于平台特定的距离来给予奖励
                    if platform_distance >= 100 and platform_distance <= RED_MISSILE_RANGE['C-400']:
                        early_intercept_reward = 1.5  # 远距离拦截额外奖励1.5
                        self.temp_reward += base_reward + early_intercept_reward
                        if self.steps > 100:  # 前100步不输出
                            print(f'[步骤{self.steps}] C-400远距离拦截奖励：距离={platform_distance:.1f}km，威胁度={platform_threat:.1f}，奖励+{base_reward + early_intercept_reward:.2f}')
                    elif platform_distance >= 70 and platform_distance < 100:
                        early_intercept_reward = 0.8  # 中远距离拦截额外奖励0.8
                        self.temp_reward += base_reward + early_intercept_reward
                    else:
                        self.temp_reward += base_reward
                # 如果defense_type是HQ-9A，并且missile_count>=2, 则给予奖励0.1
                elif defense_type =='HQ-9A' and missile_count >= 2:
                    # print(f'执行{defense_type}，打出{missile_count}枚，得分0.1')
                    self.temp_reward += 0.1
                elif defense_type == 'HQ-12' and missile_count >= 2:
                    self.temp_reward += 0.1
                    # print(f'执行{defense_type}，打出{missile_count}枚，得分0.1')
                # 当空中威胁数量很多时，鼓励HQ-9A/HQ-12主动协同拦截
                current_threats = len(self.side.contacts)
                if defense_type in ['HQ-9A', 'HQ-12'] and current_threats >= self.mass_threat_bonus_threshold and missile_count > 0:
                    self.temp_reward += self.mass_threat_bonus

                # 距离型奖励塑形：改进版，基于平台特定的距离来给予奖励
                near_thresh = self.fire_near_ratio * max_range
                far_thresh = self.fire_far_ratio * max_range
                early_thresh = self.early_intercept_distance * max_range
                
                # 使用平台特定的距离
                if platform_distance < 999.0:  # 有效距离
                    # 关键改进：C-400远距离拦截给予大幅奖励
                    if defense_type == 'C-400':
                        # C-400最优拦截距离：100-180km（远距离早期拦截）
                        if platform_distance >= 100 and platform_distance <= max_range:
                            self.temp_reward += 2.0  # 远距离拦截大幅奖励
                        # C-400中远距离：70-100km（也鼓励拦截，但奖励稍低）
                        elif platform_distance >= 70 and platform_distance < 100:
                            self.temp_reward += 1.0
                        # C-400近距离：<70km（已经比较近了，奖励较少）
                        elif platform_distance < 70:
                            self.temp_reward += 0.3
                    else:
                        # HQ-9A和HQ-12的距离奖励逻辑
                        # 早期拦截奖励（距离较远但在射程内，给予额外奖励）
                        if platform_distance >= early_thresh and platform_distance <= max_range:
                            self.temp_reward += self.early_intercept_bonus * 0.5  # 早期拦截额外奖励
                        # 中距离拦截奖励（最佳拦截距离）
                        elif platform_distance <= near_thresh:
                            self.temp_reward += self.fire_reward_near
                        # 过远拦截惩罚（可能浪费导弹）
                        elif platform_distance >= far_thresh:
                            self.temp_reward -= self.fire_penalty_far
                
                # 4. 执行手动打击（人工分配调用）
                # 从单元的实际武器列表中获取正确的weapon_dbid字符串
                weapon_list = self._get_unit_weapon(best_unit_dict['unit'])
                weapon_id = RED_MISSILE_INFO_MAP[defense_type][0]
                
                # 从武器列表中查找匹配的武器字符串（支持123和2000002）
                # weapon_list格式：[['数量', '武器名称', '武器字符串']]
                weapon_type_ids = []
                if defense_type == 'HQ-12':
                    weapon_type_ids = [123, 2000002]  # 支持两种DBID
                elif defense_type == 'HQ-9A':
                    weapon_type_ids = [1225]
                elif defense_type == 'C-400':
                    weapon_type_ids = [2104]
                
                # 从武器列表中查找匹配的武器
                weapon_dbid_to_use = None
                for weapon_item in weapon_list:
                    if len(weapon_item) >= 3:
                        weapon_str = weapon_item[-1]  # 武器字符串（最后一个元素）
                        # 从武器字符串中提取数字ID
                        try:
                            extracted_id = int(re.sub('\D', '', weapon_str))
                            if extracted_id in weapon_type_ids:
                                weapon_dbid_to_use = weapon_str
                                break  # 找到匹配的武器，使用它
                        except:
                            continue
                
                # 如果从武器列表中没有找到，尝试使用配置的weapon_dbid（向后兼容）
                if weapon_dbid_to_use is None and weapon_id in self.weapon_dbid:
                    weapon_dbid_to_use = self.weapon_dbid[weapon_id]
                
                # 如果还是没有找到，打印错误并跳过
                if weapon_dbid_to_use is None:
                    print(f'[步骤{self.steps}] ⚠️ 错误：无法找到{defense_type}的weapon_dbid（武器类型ID={weapon_type_ids}），跳过此次拦截')
                    print(f'[步骤{self.steps}] 调试：单元={best_unit_dict["unit"].strName}, 武器列表={weapon_list}')
                    continue
                
                success = _manual_attack(
                    unit=best_unit_dict['unit'],
                    target=target,
                    missile_count=best_unit_dict['missile_count'],
                    weapon_dbid=weapon_dbid_to_use,
                    platform_type=defense_type,  # 平台类型参数
                    step=self.steps  # 传递步骤数用于输出日志
                )

                # 5. 更新单元发射记录与总体弹药（用于留弹意识和成本）
                if success:
                    unit_guid = best_unit_dict['unit'].strGuid
                    if unit_guid not in self.unit_reload_time:
                        self.unit_reload_time[unit_guid] = {
                            'last_fire_step': self.steps,
                            'defense_type': defense_type
                        }
                    else:
                        self.unit_reload_time[unit_guid]['last_fire_step'] = self.steps
                    # 更新整体弹药估计与本步发射统计
                    fired = best_unit_dict['missile_count']
                    if defense_type in self.missile_inventory:
                        self.missile_inventory[defense_type] = max(
                            0, self.missile_inventory[defense_type] - fired
                        )
                    step_fire_stats[defense_type] += fired

                # 6. 记录拦截（无论是否成功，后续用来判定击落奖励）
                executed_engagements.append({
                    'target_id': target_id,
                    'defense_type': defense_type,
                    'unit_id': best_unit_dict['unit'].strGuid,
                    'missile_count': best_unit_dict['missile_count'],
                    'distance': target_distance,
                    'time': self.m_Time,
                    'success': success,
                })

            # 注意：冷却机制已完全移除，不再记录engagement_cooldown

        # 更新最近的拦截记录（用于奖励计算）
        self.recent_engagements = executed_engagements

        # ===== 2.1 导弹成本奖励：根据本步发射的导弹数量施加成本惩罚 =====
        self.last_step_fire_stats = step_fire_stats
        total_cost = 0.0
        for dt, num in step_fire_stats.items():
            if num <= 0:
                continue
            unit_cost = RED_MISSILE_COST.get(dt, 0)
            total_cost += unit_cost * num
        if total_cost > 0:
            self.temp_reward -= self.missile_cost_alpha * total_cost

        # ===== 2.3 多平台协同与不过度堆弹奖励/惩罚 =====
        # 以目标为粒度统计本步各平台对同一目标的发射情况
        target_fire = {}
        for eng in executed_engagements:
            tid = eng['target_id']
            dt = eng['defense_type']
            cnt = eng['missile_count']
            if tid not in target_fire:
                target_fire[tid] = {'total': 0, 'types': set()}
            target_fire[tid]['total'] += cnt
            if cnt > 0:
                target_fire[tid]['types'].add(dt)

        for tid, info in target_fire.items():
            total = info['total']
            types = info['types']
            # 协同奖励：C-400 与 HQ 系列共同拦截，且总发射量在2~4之间
            if 'C-400' in types and (('HQ-9A' in types) or ('HQ-12' in types)):
                if 2 <= total <= 4:
                    self.temp_reward += 0.5
            # 过度堆弹惩罚：对单目标短时间发射过多导弹
            if total >= 7:
                self.temp_reward -= 0.5

        return executed_engagements

    def _select_best_unit_for_engagement(self, defense_type, target, missile_count):
        """
        为拦截任务选择最优单元

        选择标准（优先级从高到低）：
        1. 单元状态正常（未损坏）
        2. 有足够的可用导弹
        3. 距离目标最近（减少飞行时间）
        4. 不在重装填状态（针对HQ-12）

        Args:
            defense_type: 防空单元类型
            target: 目标对象
            missile_count: 需要的导弹数量

        Returns:
            最优单元对象，如果没有可用单元则返回None
        """
        # 获取该类型的所有单元
        if defense_type == 'C-400':
            candidates = [v for k, v in self.side.facilities.items() if 'C-400' in v.strName]
        elif defense_type == 'HQ-9A':
            candidates = [v for k, v in self.side.facilities.items() if 'HQ-9A' in v.strName]
        else:  # HQ-12
            candidates = [v for k, v in self.side.facilities.items() if '红旗-12' in v.strName or 'HQ-12' in v.strName]

        # 调试信息：检查candidates是否为空
        if not candidates:
            print(f'[步骤{self.steps}] ⚠️ 调试：{defense_type}的candidates为空，无法找到该类型的单元')
            # 输出所有设施名称以便排查
            all_facility_names = [v.strName for k, v in self.side.facilities.items()]
            print(f'[步骤{self.steps}] 当前所有设施名称：{all_facility_names[:10]}...（最多显示10个）')
            return None

        # 筛选可用单元
        valid_units = []
        skipped_reasons = {'no_missiles': 0, 'other': 0}
        
        for unit in candidates:
            unit_name = unit.strName
            # 检查导弹数量
            weapon_list = self._get_unit_weapon(unit)
            # 武器类型列表：包含123（旧DBID）和2000002（新DBID），以兼容不同想定
            available_missiles = self._get_weapon_num(weapon_list, [123, 2000002, 1225, 2104])

            # if defense_type == 'C-400':
            #     available_missiles = available_missiles - 32  # 32枚需要重装载

            if defense_type == 'HQ-12':
                # available_missiles = available_missiles - 24
                pass

            if available_missiles == 0:
                skipped_reasons['no_missiles'] += 1
                continue  # 导弹不足, 不讨论导弹不足问题，C-400也有重加载问题
            
            # 如果可用数量少于动作空间生成的数量，那么就把所有的都发射出去
            actual_missile_count = missile_count
            if available_missiles < missile_count:
                actual_missile_count = available_missiles
            
            # 计算到目标的距离，如果选择最近距离，虽然反应快，但是最后结果不一定好。
            distance = get_two_point_distance(unit.dLatitude, unit.dLongitude, target.dLatitude, target.dLongitude)
            # 综合评分：综合考虑距离、可用导弹数量、单元状态
            # 评分 = 导弹充足度(0-1) * 0.5 + 距离近度(0-1) * 0.5
            missile_score = available_missiles / 50 # 导弹越多越好
            distance_score = max(0.0, (100 - distance) / 100)  # 距离越近越好
            combined_score = missile_score * 0.5 + distance_score * 0.5
            
            valid_units.append({
                'unit': unit,
                'combined_score': combined_score,
                'missile_count': actual_missile_count,
            })

        # 调试信息：如果valid_units为空，输出详细原因
        if not valid_units:
            print(f'[步骤{self.steps}] ⚠️ 警告：{defense_type}的valid_units为空！')
            print(f'[步骤{self.steps}] 调试：找到{len(candidates)}个{defense_type}候选单元：{[c.strName for c in candidates]}')
            print(f'[步骤{self.steps}] 调试：跳过原因统计={skipped_reasons}')
            # 详细输出每个候选单元的检查结果
            for unit in candidates:
                unit_name = unit.strName
                # 检查单元武器字符串
                unit_weapons_str = getattr(unit, 'm_UnitWeapons', 'N/A')
                print(f'[步骤{self.steps}] 调试详情：单元={unit_name}, m_UnitWeapons字符串={unit_weapons_str[:200] if unit_weapons_str != "N/A" else "N/A"}...')
                
                try:
                    weapon_list = self._get_unit_weapon(unit)
                    weapon_list_len = len(weapon_list) if weapon_list else 0
                    # 输出武器列表的详细信息
                    if weapon_list:
                        print(f'[步骤{self.steps}] 调试详情：单元={unit_name}, 武器列表长度={weapon_list_len}, 武器列表前3项={weapon_list[:3]}')
                    else:
                        print(f'[步骤{self.steps}] 调试详情：单元={unit_name}, 武器列表为空！')
                    
                    # 检查每种武器类型的匹配情况（包含123和2000002以兼容不同想定）
                    weapon_type_list = [123, 2000002, 1225, 2104]
                    for wt in weapon_type_list:
                        wt_name = {123: 'HQ-12(旧)', 2000002: 'HQ-12(新)', 1225: 'HQ-9A', 2104: 'C-400'}.get(wt, str(wt))
                        count = self._get_weapon_num(weapon_list, [wt])
                        if count > 0:
                            print(f'[步骤{self.steps}] 调试详情：单元={unit_name}, {wt_name}(DBID={wt})武器数量={count}')
                    
                    available_missiles = self._get_weapon_num(weapon_list, weapon_type_list)
                    print(f'[步骤{self.steps}] 调试详情：单元={unit_name}, 总可用导弹数={available_missiles}（从武器类型{[123, 2000002, 1225, 2104]}统计）')
                except Exception as e:
                    print(f'[步骤{self.steps}] 调试详情：单元={unit_name}, 获取武器信息时发生异常：{e}')
            return None

        best = max(valid_units, key=lambda x: x['combined_score'])
        return best
    
    def _find_alternative_unit(self, defense_type, target, missile_count, exclude_guid):
        """
        查找替代单元（用于当首选单元在装弹时）
        
        Args:
            defense_type: 防空单元类型
            target: 目标对象
            missile_count: 需要的导弹数量
            exclude_guid: 要排除的单元GUID
        
        Returns:
            替代单元字典，如果没有则返回None
        """
        # 获取该类型的所有单元
        if defense_type == 'C-400':
            candidates = [v for k, v in self.side.facilities.items() if 'C-400' in v.strName]
        elif defense_type == 'HQ-9A':
            candidates = [v for k, v in self.side.facilities.items() if 'HQ-9A' in v.strName]
        else:  # HQ-12
            candidates = [v for k, v in self.side.facilities.items() if '红旗-12' in v.strName or 'HQ-12' in v.strName]
        
        if not candidates:
            return None
        
        # 筛选可用单元（排除指定的单元）
        valid_units = []
        for unit in candidates:
            if unit.strGuid == exclude_guid:
                continue  # 排除指定单元
            
            # 检查单元是否在装弹
            unit_info = self.unit_reload_time.get(unit.strGuid)
            if unit_info:
                last_fire_step = unit_info.get('last_fire_step', 0)
                missiles_remaining = unit_info.get('missiles_remaining', 0)
                reload_duration = self.unit_reload_duration.get(defense_type, 6)
                
                if missiles_remaining <= 0 and (self.steps - last_fire_step) < reload_duration:
                    continue  # 该单元正在装弹
            
            # 检查导弹数量
            weapon_list = self._get_unit_weapon(unit)
            # 武器类型列表：包含123（旧DBID）和2000002（新DBID），以兼容不同想定
            available_missiles = self._get_weapon_num(weapon_list, [123, 2000002, 1225, 2104])
            if defense_type == 'C-400':
                available_missiles = available_missiles - 32  # 32枚需要重装载
            if defense_type == 'HQ-12':
                # available_missiles = available_missiles - 24
                pass
            
            if available_missiles == 0:
                continue
            
            if available_missiles < missile_count:
                missile_count = available_missiles
            
            distance = get_two_point_distance(unit.dLatitude, unit.dLongitude, target.dLatitude, target.dLongitude)
            missile_score = available_missiles / 50
            distance_score = max(0.0, (100 - distance) / 100)
            combined_score = missile_score * 0.5 + distance_score * 0.5
            
            valid_units.append({
                'unit': unit,
                'combined_score': combined_score,
                'missile_count': missile_count,
            })
        
        if not valid_units:
            return None
        
        best = max(valid_units, key=lambda x: x['combined_score'])
        return best

    def _get_detected_missiles_sorted(self):
        """
        获取当前探测到的蓝方导弹（不再统一排序，因为现在威胁度是基于每个平台类型计算的）

        Returns:
            list: [(target_id, target_obj), ...] 返回所有探测到的导弹列表
        """
        # 获取所有探测到的蓝方导弹
        # m_ContactType == 1 表示导弹类型
        detected = {k: v for k, v in self.side.contacts.items() if v.m_ContactType == 1}

        # 直接返回所有探测到的导弹列表（不再统一排序）
        return [(target_id, target) for target_id, target in detected.items()]

    def _calculate_threat_score(self, missile):
        """
        计算单个导弹的威胁分数（旧版本，保留以兼容其他代码）
        注意：现在应该使用 _calculate_threat_score_for_platform 来计算基于特定平台的威胁度

        Args:
            missile: 导弹对象

        Returns:
            float: 威胁分数（0-15，范围扩大以更好区分优先级）
        """
        # 使用到保护目标的距离作为默认计算方式
        min_distance = self._get_distance_to_protected_targets(missile)
        distance_score = max(0.0, (100 - min_distance) / 100) * 5.0  # 0-5分

        speed = missile.fCurrentSpeed if hasattr(missile, 'fCurrentSpeed') else 0
        speed_score = min(speed / 1000.0, 1.0) * 2.0  # 0-2分

        total_score = distance_score + speed_score
        return total_score

    def _calculate_threat_score_for_platform(self, missile, defense_type):
        """
        计算单个导弹对特定平台类型的威胁分数（基于该平台的距离）

        Args:
            missile: 导弹对象
            defense_type: 平台类型（'C-400'/'HQ-9A'/'HQ-12'）

        Returns:
            float: 威胁分数（0-15，范围扩大以更好区分优先级）
        """
        # 1. 距离因子（距离该平台类型越近，威胁越大）
        min_distance = self._get_distance_to_platform_type(missile, defense_type)
        max_range = RED_MISSILE_RANGE[defense_type]
        
        # 距离评分：在射程内，距离越近分数越高；超出射程则分数较低
        if min_distance <= max_range:
            # 在射程内：距离越近分数越高（归一化到0-5分）
            distance_score = max(0.0, (max_range - min_distance) / max_range) * 5.0  # 0-5分
        else:
            # 超出射程：给一个较小的分数（基于距离）
            distance_score = max(0.0, (200 - min_distance) / 200) * 1.0  # 0-1分

        # 2. 速度因子（速度越快，留给拦截的时间越短）
        speed = missile.fCurrentSpeed if hasattr(missile, 'fCurrentSpeed') else 0
        speed_score = min(speed / 1000.0, 1.0) * 2.0  # 0-2分

        # 3. 可拦截性因子（在射程内给予更高分数）
        in_range_score = 0.0
        if min_distance <= max_range:
            # 在射程内，根据距离给出不同的分数
            if defense_type == 'C-400':
                if min_distance >= 100 and min_distance <= max_range:  # 100-180km，C-400最优拦截距离
                    in_range_score = 3.0
                elif min_distance >= 70 and min_distance < 100:  # 70-100km
                    in_range_score = 2.5
                elif min_distance < 70:  # <70km
                    in_range_score = 2.0
            elif defense_type == 'HQ-9A':
                if min_distance >= 50 and min_distance <= max_range:  # 50-100km，HQ-9A最优拦截距离
                    in_range_score = 2.5
                elif min_distance < 50:  # <50km
                    in_range_score = 2.0
            else:  # HQ-12
                if min_distance <= max_range:  # 在射程内（≤70km）
                    in_range_score = 2.0

        # 4. 时间紧迫度（预计到达该平台的时间）
        if speed > 0 and min_distance < 999.0:
            eta = min_distance / speed  # 预计到达时间（小时）
            urgency_score = max(0, (1 - eta / 0.3)) * 2.0  # 18分钟内到达则分数高
        else:
            urgency_score = 0.0

        # 5. 早期拦截奖励因子（鼓励在远距离拦截）
        early_intercept_bonus = 0.0
        if defense_type == 'C-400' and min_distance >= 100 and min_distance <= max_range:
            # C-400远距离拦截范围（100-180km）
            early_intercept_bonus = 2.0  # 额外2分，鼓励C-400远距离拦截
        elif defense_type == 'C-400' and min_distance >= 70 and min_distance < 100:
            # C-400中远距离（70-100km）
            early_intercept_bonus = 1.0
        elif defense_type == 'HQ-9A' and min_distance >= 60 and min_distance <= max_range:
            # HQ-9A中远距离拦截
            early_intercept_bonus = 1.0

        total_score = distance_score + speed_score + in_range_score + urgency_score + early_intercept_bonus
        return total_score

    def _get_distance_to_protected_targets(self, missile):
        """
        计算导弹到保护目标的距离(保护目标的平均经纬度)

        Args:
            missile: 导弹对象

        Returns:
            float: 最小距离（km）
        """

        if not self.protected_target:
            return 999.0  # 如果没有保护目标，返回一个大值

        dist = get_two_point_distance(
            missile.dLatitude, missile.dLongitude,
            self.target_middle_point[0], self.target_middle_point[1]
        )

        return dist

    def _get_distance_to_platform_type(self, missile, defense_type):
        """
        计算导弹到指定平台类型的最短距离（在所有该类型的平台中选择最近的）

        Args:
            missile: 导弹对象
            defense_type: 平台类型（'C-400'/'HQ-9A'/'HQ-12'）

        Returns:
            float: 到该平台类型的最短距离（km），如果没有该类型平台则返回999.0
        """
        # 获取该类型的所有平台
        if defense_type == 'C-400':
            platforms = [v for k, v in self.side.facilities.items() if 'C-400' in v.strName]
        elif defense_type == 'HQ-9A':
            platforms = [v for k, v in self.side.facilities.items() if 'HQ-9A' in v.strName]
        elif defense_type == 'HQ-12':
            platforms = [v for k, v in self.side.facilities.items() if '红旗-12' in v.strName or 'HQ-12' in v.strName]
        else:
            return 999.0

        if not platforms:
            return 999.0

        # 计算到所有该类型平台的距离，返回最小值
        min_dist = float('inf')
        for platform in platforms:
            dist = get_two_point_distance(
                missile.dLatitude, missile.dLongitude,
                platform.dLatitude, platform.dLongitude
            )
            min_dist = min(min_dist, dist)

        return min_dist if min_dist != float('inf') else 999.0

    def _apply_rule_based_correction(self, engagement_plan):
        """
        基于规则的导弹发射修正函数
        
        根据距离、剩余弹药、时间进度等规则，修正智能体的拦截计划
        
        Args:
            engagement_plan: 原始拦截计划列表
        
        Returns:
            corrected_plan: 修正后的拦截计划列表
            step_fire_stats: 本步各类导弹发射统计（用于全局上限检查）
        """
        if not self.enable_rule_based_correction:
            return engagement_plan, {'C-400': 0, 'HQ-9A': 0, 'HQ-12': 0}
        
        corrected_plan = []
        step_fire_stats = {'C-400': 0, 'HQ-9A': 0, 'HQ-12': 0}
        
        # 计算时间进度
        time_delta = self.m_Time - self.m_StartTime
        time_frac = max(0.0, min(1.0, time_delta / 5400.0))  # 5400秒 = 1.5小时
        
        # 计算剩余弹药比例
        remaining_ratios = {}
        for k, v in self.missile_inventory.items():
            init_v = self.initial_missile_inventory.get(k, v)
            remaining_ratios[k] = v / init_v if init_v > 0 else 1.0
        
        # 计算目标速度（用于HQ-9A发射时机判断）
        def get_target_speed(target):
            return target.fCurrentSpeed if hasattr(target, 'fCurrentSpeed') else 0
        
        # 计算ETA（预计到达时间，小时）
        def calculate_eta(distance, speed):
            if speed > 0 and distance < 999.0:
                return distance / speed  # 距离(km) / 速度(km/h)
            return 999.0
        
        # 对每个拦截计划进行规则修正
        for plan in engagement_plan:
            target = plan['target']
            assignments = plan.get('defense_assignments', {}).copy()
            platform_distances = plan.get('platform_distances', {})
            platform_threats = plan.get('platform_threats', {})
            
            # 获取目标信息
            target_speed = get_target_speed(target)
            
            # ===== C-400规则修正 =====
            c400_count = assignments.get('C-400', 0)
            c400_distance = platform_distances.get('C-400', 999.0)
            c400_threat = platform_threats.get('C-400', 0.0)
            c400_remaining_ratio = remaining_ratios.get('C-400', 1.0)
            target_distance = plan.get('target_distance', 999.0)  # 蓝方导弹到保护目标的距离
            
            # 判断是否威胁到C-400本身（距离C-400平台很近）
            is_threat_to_c400 = c400_distance < 60  # 距离C-400平台<60km，说明威胁到C-400本身
            
            if c400_count > 0 or is_threat_to_c400:
                # 规则1: 基于距离的分层发射策略
                if is_threat_to_c400:
                    # 距离C-400平台 < 60km：必须拦截自卫（否则C-400会被摧毁）
                    c400_count = max(c400_count, 2) if c400_count > 0 else 2  # 至少发射2枚自卫
                    if self.steps > 100:
                        print(f'[规则修正] C-400: 目标距离C-400平台{c400_distance:.1f}km < 60km，必须拦截自卫，发射{c400_count}枚')
                elif c400_distance > self.c400_max_distance:
                    # 距离 > 150km：不发射（留待更近时）
                    c400_count = 0
                    if self.steps > 100:
                        print(f'[规则修正] C-400: 目标距离{c400_distance:.1f}km > {self.c400_max_distance}km，取消发射')
                elif c400_distance >= 120 and c400_distance <= 150:
                    # 距离 120-150km：发射1枚（高威胁目标可2枚）
                    if c400_threat >= 8.0:
                        c400_count = min(c400_count, 2) if c400_count > 0 else 1
                    else:
                        c400_count = min(c400_count, 1) if c400_count > 0 else 0
                elif c400_distance >= 100 and c400_distance < 120:
                    # 距离 100-120km：发射1-2枚（根据威胁度）
                    c400_count = min(c400_count, 2) if c400_count > 0 else 1
                elif c400_distance >= 70 and c400_distance < 100:
                    # 距离 70-100km：发射1枚（优先让HQ-9A拦截，但C-400也可以辅助）
                    c400_count = min(c400_count, 1) if c400_count > 0 else 0
                elif c400_distance >= 60 and c400_distance < 70:
                    # 距离 60-70km：发射1-2枚（中等威胁，可以拦截）
                    c400_count = min(c400_count, 2) if c400_count > 0 else 1
                
                # 规则2: 基于剩余弹药的动态调整（但自卫拦截不受此限制）
                if not is_threat_to_c400:  # 只有在非自卫情况下才限制
                    if c400_remaining_ratio > 0.5:
                        # 剩余弹药 > 50%：按上述规则执行
                        pass
                    elif c400_remaining_ratio >= 0.3:
                        # 剩余弹药 30-50%：仅在距离 > 120km 且威胁度 ≥ 8.0 时发射，或距离<60km自卫
                        if not (c400_distance >= 120 and c400_threat >= 8.0):
                            c400_count = 0
                            if self.steps > 100:
                                print(f'[规则修正] C-400: 剩余弹药比例{c400_remaining_ratio:.1%}，仅在远距离高威胁时发射')
                    else:
                        # 剩余弹药 < 30%：仅在距离 > 140km 且威胁度 ≥ 9.0 时发射，或距离<60km自卫，且单目标最多1枚
                        if not (c400_distance >= 140 and c400_threat >= 9.0):
                            c400_count = 0
                            if self.steps > 100:
                                print(f'[规则修正] C-400: 剩余弹药比例{c400_remaining_ratio:.1%}，仅在极远距离极高威胁时发射')
                        else:
                            c400_count = min(c400_count, 1)
                
                # 规则3: 基于时间进度的弹药保留（但自卫拦截不受此限制）
                if time_frac > self.late_phase and not is_threat_to_c400:
                    # 时间进度 > 70%：保留至少20%弹药（约32枚），但自卫拦截不受限制
                    reserve_count = int(self.initial_missile_inventory.get('C-400', 160) * self.c400_reserve_ratio)
                    if self.missile_inventory.get('C-400', 0) <= reserve_count:
                        c400_count = 0
                        if self.steps > 100:
                            print(f'[规则修正] C-400: 后期阶段，保留弹药，取消发射（非自卫情况）')
                
                # 规则4: 单步全局发射上限（在最后检查）
                current_c400_fire = step_fire_stats.get('C-400', 0)
                if current_c400_fire + c400_count > self.c400_step_max:
                    c400_count = max(0, self.c400_step_max - current_c400_fire)
                    if self.steps > 100 and c400_count == 0 and assignments.get('C-400', 0) > 0:
                        print(f'[规则修正] C-400: 本步已发射{current_c400_fire}枚，达到单步上限{self.c400_step_max}枚，取消本次发射')
            
            assignments['C-400'] = c400_count
            if c400_count > 0:
                step_fire_stats['C-400'] += c400_count
            
            # ===== HQ-9A规则修正 =====
            hq9a_count = assignments.get('HQ-9A', 0)
            hq9a_distance = platform_distances.get('HQ-9A', 999.0)
            hq9a_threat = platform_threats.get('HQ-9A', 0.0)
            hq9a_remaining_ratio = remaining_ratios.get('HQ-9A', 1.0)
            hq9a_eta = calculate_eta(hq9a_distance, target_speed)
            
            if hq9a_count > 0 or hq9a_distance < 80:  # 即使未分配，如果距离很近也需要检查
                # 规则1: 基于距离的发射时机
                if hq9a_distance > self.hq9a_max_distance:
                    # 距离 > 100km：不发射（超出射程）
                    hq9a_count = 0
                elif hq9a_distance >= 80 and hq9a_distance <= 100:
                    # 距离 80-100km：仅在威胁度 ≥ 7.0 时发射1-2枚（早期拦截窗口）
                    if hq9a_threat >= 7.0:
                        hq9a_count = min(hq9a_count, 2) if hq9a_count > 0 else 1
                    else:
                        hq9a_count = 0
                elif hq9a_distance >= self.hq9a_optimal_distance_min and hq9a_distance <= self.hq9a_optimal_distance_max:
                    # 距离 60-80km：发射2枚（最佳拦截距离）
                    hq9a_count = max(hq9a_count, 2) if hq9a_count > 0 else 2
                elif hq9a_distance >= 40 and hq9a_distance < 60:
                    # 距离 40-60km：发射2-3枚（必须拦截）
                    hq9a_count = max(hq9a_count, 2) if hq9a_count > 0 else 2
                    hq9a_count = min(hq9a_count, 3)
                elif hq9a_distance < 40:
                    # 距离 < 40km：发射3枚（紧急拦截）
                    hq9a_count = max(hq9a_count, 3) if hq9a_count > 0 else 3
                
                # 规则2: 基于目标速度的发射时机调整
                if hq9a_count > 0:
                    if target_speed > 800:  # 速度 > 800 m/s
                        # 提前发射（在距离80-100km时开始拦截）
                        if hq9a_distance >= 80 and hq9a_distance <= 100:
                            hq9a_count = max(hq9a_count, 1) if hq9a_count > 0 else 1
                    elif target_speed < 600:  # 速度 < 600 m/s
                        # 可稍晚发射（在距离40-60km时拦截）
                        if hq9a_distance > 60:
                            hq9a_count = 0  # 距离还远，不发射
                
                # 规则3: 基于ETA的发射时机
                if hq9a_count > 0 and hq9a_eta < 999.0:
                    if hq9a_eta > 5.0 / 60:  # ETA > 5分钟
                        # 不发射（等待更近时）
                        hq9a_count = 0
                    elif hq9a_eta >= 3.0 / 60 and hq9a_eta <= 5.0 / 60:  # ETA 3-5分钟
                        # 发射1-2枚（早期拦截）
                        hq9a_count = min(hq9a_count, 2)
                    elif hq9a_eta >= 1.0 / 60 and hq9a_eta < 3.0 / 60:  # ETA 1-3分钟
                        # 发射2-3枚（必须拦截）
                        hq9a_count = max(hq9a_count, 2)
                        hq9a_count = min(hq9a_count, 3)
                    elif hq9a_eta < 1.0 / 60:  # ETA < 1分钟
                        # 发射3枚（紧急拦截）
                        hq9a_count = max(hq9a_count, 3)
                
                # 规则4: 避免过早发射的检查
                if hq9a_count > 0:
                    if hq9a_distance > 90 and target_speed < 600:
                        # 目标距离 > 90km 且速度 < 600 m/s：延迟发射
                        hq9a_count = 0
                        if self.steps > 100:
                            print(f'[规则修正] HQ-9A: 目标距离{hq9a_distance:.1f}km且速度{target_speed:.0f}m/s，延迟发射')
                    elif hq9a_distance < 50 and target_speed > 800:
                        # 目标距离 < 50km 且速度 > 800 m/s：立即发射
                        hq9a_count = max(hq9a_count, 3)
                
                # 规则5: 基于时间进度的弹药保留
                if time_frac > self.late_phase:
                    # 时间进度 > 70%：保留至少15%弹药（约30枚）
                    reserve_count = int(self.initial_missile_inventory.get('HQ-9A', 192) * self.hq9a_reserve_ratio)
                    if self.missile_inventory.get('HQ-9A', 0) <= reserve_count and hq9a_distance > 60:
                        # 只有在距离较远时才保留，近距离必须拦截
                        hq9a_count = 0
                        if self.steps > 100:
                            print(f'[规则修正] HQ-9A: 后期阶段且距离较远，保留弹药')
            
            assignments['HQ-9A'] = hq9a_count
            if hq9a_count > 0:
                step_fire_stats['HQ-9A'] += hq9a_count
            
            # ===== 综合协调规则：平台优先级分配 =====
            # 如果C-400和HQ-9A都分配了，根据距离调整优先级（但自卫拦截不受此限制）
            if assignments.get('C-400', 0) > 0 and assignments.get('HQ-9A', 0) > 0:
                if is_threat_to_c400:
                    # 如果威胁到C-400本身，C-400必须拦截自卫，不取消
                    # 但HQ-9A也可以协同拦截
                    pass
                elif c400_distance >= 100:
                    # 距离 > 100km：仅C-400拦截
                    old_hq9a = assignments['HQ-9A']
                    assignments['HQ-9A'] = 0
                    step_fire_stats['HQ-9A'] -= old_hq9a
                    hq9a_count = 0
                elif c400_distance < 60:
                    # 距离 < 60km：C-400自卫，HQ-9A也可以协同
                    # 不取消C-400，因为这是自卫拦截
                    pass
                elif c400_distance >= 60 and c400_distance < 70:
                    # 距离 60-70km：HQ-9A为主，C-400可以取消（非自卫情况）
                    if not is_threat_to_c400:
                        old_c400 = assignments['C-400']
                        assignments['C-400'] = 0
                        step_fire_stats['C-400'] -= old_c400
                        c400_count = 0
            
            # 更新修正后的拦截计划
            corrected_plan.append({
                **plan,
                'defense_assignments': assignments
            })
        
        return corrected_plan, step_fire_stats





