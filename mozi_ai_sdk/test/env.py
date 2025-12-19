#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import cos
from math import radians
from mozi_ai_sdk.test import etc

from mozi_ai_sdk.base_env import BaseEnvironment as base_env
import numpy as np
from mozi_utils import pylog
from mozi_utils.geo import get_point_with_point_bearing_distance
from mozi_utils.geo import get_degree
from mozi_utils.geo import get_two_point_distance
import math
import os


class Antiship(base_env):
    """
    环境类
    """
    def __init__(self,
                 server_ip,
                 sever_port,
                 agent_key_event_file=None,
                 duration_interval=None,
                 app_mode=None,
                 synchronous=None,
                 simulate_compression=None,
                 scenario_name=None,
                 platform_mode=None,
                 platform="windows"):
        super().__init__(server_ip,
                         sever_port,
                         platform,
                         scenario_name,
                         simulate_compression,
                         duration_interval,
                         synchronous,
                         app_mode,
                         platform_mode)

        self.SERVER_PLAT = platform
        self.agent_key_event_file = agent_key_event_file

        self.PI = 3.1415926535897932
        self.degree2radian = self.PI / 180.0

        self.blue_side_name = "蓝方"
        self.red_side_name = "红方"

        """
        敌我双方船和dd 字典形式
        """
        self.blue_ship = {}
        self.blue_missile = {}
        self.red_ship = {}
        self.red_missile = {}
        self.red_enemies = {}
        self.rank_dic = {}                          # 什么意思 作用是什么

        self.n_agents = 1                                                   # 可以不用？
        self.n_red_ship = 1
        self.n_red_missile = 30
        self.n_enemies = self.n_red_missile+self.n_red_ship                             # 敌方数量是船的数量＋弹的数量
        self.n_actions = self.n_enemies * 3 + 1
        self.step_limit = 16

        #self.death_tracker_ally= [0] * self.n_agents
        self.agent_is_dead = [0]                                             # 蓝方船是否阵亡
        self.prev_health = [1] * self.n_agents
        self.now_health = [1] * self.n_agents

        self.red_ship_is_dead = [0]                                         # 红方船是否阵亡
        self.red_prev_health = [1] * self.n_red_ship
        self.red_now_health = [1] * self.n_red_ship

        self.prev_mount = [10, 10, 10]                   # 蓝方船初始弹量
        self.all_avail_target = [0] * self.n_enemies     # 目标数

        #self.reward_
        #self.reward_
        self.reward_win = 1
        self.reward_defeat = -1

        self.step = 0

        """
        舰空（sta），舰舰（sts） 按顺序分别为武器的火力范围
        """
        self.sta_fire_range_1 = 92600
        self.sta_min_fire_range_1 = 3704
        self.sts_fire_range_1 = 46300
        self.sts_min_fire_range_1 = 3704

        self.sta_fire_range_2 = 92600
        self.sta_min_fire_range_2 = 3704
        self.sts_fire_range_2 = 46300
        self.sts_min_fire_range_2 = 3704

        self.sta_fire_range_3 = 0
        self.sta_min_fire_range_3 = 0
        self.sts_fire_range_3 = 2499999
        self.sts_min_fire_range_3 = 2499999

        self.weapon_guid = ['hsfw-dataweapon-00000000001195','hsfw-dataweapon-00000000001194',
                            'hsfw-dataweapon-00000001003786']
        # 顺序为 标2-2、标2-5、战斧-3000

        self.all_impact_dist = 0
        self.target = []

    def reset(self, app_mode=None):
        """
        重置
        """
        super(Antiship, self).reset()
        self.blue_ship = {}
        self.blue_missile = {}
        self.red_ship = {}
        self.red_missile = {}
        self.red_enemies = {}
        self.red_enemies = self.red_ship.copy()
        self.red_enemies.update(self.red_missile)
        self.rank_dic = {}

        # self.death_tracker_ally = [0] * self.n_agents
        self.agent_is_dead = [0]
        self.prev_health = [1] * self.n_agents
        self.now_health = [1] * self.n_agents

        self.prev_mount = [10, 10, 10]                   # 蓝方船初始弹量
        self.all_avail_target = [0] * self.n_enemies     # 目标数
        self.step = 0

        self.all_impact_dist = 0
        self.target = []

        self.scenario.set_cur_side_and_dir_view("蓝方", "false")    # 设置当前推演方及是否显示导演视图
        self._construct_side_entity()   #建立实体对象
        self._init_unit_list()

        self.scenario = super(Antiship, self).step()

    def _construct_side_entity(self):
        """
        构造各方实体，该函数无问题
        """
        self.redside = self.scenario.get_side_by_name(self.red_side_name)
        self.redside.static_construct()
        self.blueside = self.scenario.get_side_by_name(self.blue_side_name)
        self.blueside.static_construct()

    def _init_unit_list(self):
        """
        初始化单元列表
        蓝船、红船、红弹

        红弹要不要初始化？
        """
        temp_value_1 = {}
        for key, value in self.blueside.ships.items():
            temp_value_1["guid"] = key
            temp_value_1["unit"] = value
            v = temp_value_1.copy()
            if self.blueside.ships[key].strName == "b1":
                blue_idx = 0
                self.center_lon = self.blueside.ships[key].dLongitude
                self.center_lat = self.blueside.ships[key].dLatitude
            else:
                blue_idx = None
            self.blue_ship[blue_idx] = v

        temp_value_2 = {}
        for key, value in self.redside.ships.items():
            temp_value_2["guid"] = key
            temp_value_2["unit"] = value
            v_ = temp_value_2.copy()
            if self.redside.ships[key].strName == "r1":
                red_idx = 0
            else:
                red_idx = None
            self.red_ship[red_idx] = v_

        temp_value_3 = {}
        for red_idx in range(self.n_red_missile):
            temp_value_3["guid"] = None
            temp_value_3["unit"] = None
            temp_value_3["expect_value"] = 1
            self.red_missile[red_idx] = temp_value_3.copy()

        self.red_enemies = self.red_ship.copy()
        self.red_enemies.update(self.red_missile)

    def get_health(self):
        """
        获取蓝方单元毁伤程度
        """
        if self.agent_is_dead[0]:
            health = 0
        else:
            ship = self.blue_ship[0]["unit"]
            damage = ship.strDamageState
            health = (100 - float(damage)) / 100
        return health

    def get_red_health(self):
        """
        获取红方单元毁伤程度
        """
        if self.red_ship_is_dead[0]:
            health = 0
        else:
            ship = self.red_ship[0]["unit"]
            damage = ship.strDamageState
            health = (100 - float(damage)) / 100
        return health

    def get_weapon_num(self):
        """
        获取蓝方单元武器总数
        """
        weapon_num_list = []
        health = self.get_health()
        if health != 0:
            ship = self.blue_ship[0]["unit"]
            total_weapon = ship.get_mounts()  # 返回dic，key是两种挂载的guid，value是两种CMount类，这里并不是数量
            for key in total_weapon:
                weapon_status = total_weapon[key].m_ComponentStatus
                if weapon_status == 0:
                    weapon_num = total_weapon[key].strLoadWeaponCount
                    index = weapon_num.find('/')
                    weapon_num = int(weapon_num[1:index])
                else:
                    weapon_num = 0
                weapon_num_list.append(weapon_num)
        else:
            weapon_num_list = [0, 0, 0]
        return weapon_num_list

    def get_avail_agent_actions(self):
        avail_actions = [1] + [0] * (self.n_actions - 1)
        health_blue = self.get_health()
        weapon_permission = [1, 1, 1]                                # 都在距离内，不加判断函数，全部设置为可以开火
        if health_blue > 0:                                                                         # 蓝船是否健康
            for e_id, e_value in enumerate(self.all_avail_target):                                 # enumerate 是一个 Python 内置函数，它用于同时遍历可迭代对象的索引和元素
                if e_value == 1:                                                                   # and e_id not in target_chosen
                    for i, j in enumerate(weapon_permission):
                        if j == 1:
                            idx = e_id + self.n_enemies * i + 1
                            avail_actions[idx] = 1
        return avail_actions

    """
    def weapon_permit(self):
        weapon_permission = [1, 1, 1]            # 初始化武器权限列表，默认都不允许开火

        return weapon_permission

    """

    def get_ship_missile_list(self):
        """该函数用于检测此时可打击的船只，和来袭导弹的数量, 全局"""
        type_list = []                                                          # 用于储存观测到的各种类型
        self.all_avail_target = [0] * self.n_enemies                            # 表示所有目标是否可用的列表

        """返回各个导弹的GUID以及实体值，所有船共同享有这两个列表"""
        for key in self.blueside.contacts:
            contact_type = self.blueside.contacts[key].m_ContactType
            type_list.append(int(contact_type))                                 # 存储所有观测的种类

            if contact_type == 1:  # 1：表示导弹
                unit = self.blueside.contacts[key]                              #
                true_unit = unit.get_actual_unit()                              # 获取实际的单位；获取目标真实单元  问题：会返回None
                if true_unit and unit.iWeaponsAimingAtMe == 0:                  # 如果实际单位存在且武器没有对准自己     作用是什么？
                    idx = self.all_avail_target.index(0)
                    # print("索引号为：", idx)
                    if idx != -1:
                        self.all_avail_target[idx] = 1
                        # print("索引号列表为：", self.all_avail_target)
                        if self.red_missile[idx]["guid"] is None:    # 防止重复计算
                            self.red_missile[idx]["guid"] = key    # 这里的key是蓝方对红方观测的key
                            self.red_missile[idx]["unit"] = true_unit

            if contact_type == 2:   # 2:表示水面目标
                unit = self.blueside.contacts[key]  #
                true_unit = unit.get_actual_unit()
                idx = self.all_avail_target.index(0)
                if idx != -1:
                    self.all_avail_target[idx] = 1
                    self.red_ship[idx]["guid"] = key  # 这里的key是蓝方对红方观测的key
                    self.red_ship[idx]["unit"] = true_unit

    def get_obs(self):
        """返回单个智能体的观测"""
        return self.get_obs_agent()

    def get_obs_agent(self):
        """
        获得智能体的观测       敌方船的信息：生命值、不同的弹量属于观测还是全局信息？、经度、维度
        """
        ship = self.blue_ship[0]["unit"]
        e_ship = self.red_ship[0]["unit"]

        """30颗敌方弹，[期望价值，夹角，距离，经度，纬度，速度，威胁度], 30x6=180      速度先不算 """
        enemy_missile_feats = np.zeros((self.n_enemies-1, 0), dtype=np.float32)              # 敌方船还未考虑

        """ 1艘敌方船,[距离，经度，纬度], 1x3=3 """
        enemy_ship_feats = np.zeros(3, dtype=np.float32)

        """自身，[经度，纬度，生命值，1弹剩余量，2弹剩余量，3弹剩余量], 1x6=6"""
        own_feats = np.zeros(6, dtype=np.float32)

        health = self.get_health()
        if health > 0:
            lon = ship.dLongitude
            lat = ship.dLatitude

            e_ship_lon = e_ship.dLongitude
            e_ship_lat = e_ship.dLatitude

            """ enemy_missile_features """

            """ enemy_ship_features"""

            enemy_ship_feats[0] = (e_ship_lon - 100) / 100
            enemy_ship_feats[1] = e_ship_lat / 100
            enemy_ship_feats[2] = get_two_point_distance(lon, lat, e_ship_lon, e_ship_lat)

            """own features"""
            own_feats[0] = (lon - 100) / 100
            own_feats[1] = lat / 100
            own_feats[2] = health

            own_weapon = self.get_weapon_num()    # 输出的是每个挂载中武器的数量，而不是各个武器的，因此我们一个挂载只搭载一种武器
            own_feats[3] = own_weapon[0] / 100
            own_feats[4] = own_weapon[1] / 100
            own_feats[5] = own_weapon[2] / 100

        agent_obs = np.concatenate(
            (
                enemy_missile_feats.flatten(),
                enemy_ship_feats.flatten(),
                own_feats.flatten()
            )
        )
        # print("观测为：", agent_obs)
        return agent_obs

    def scenario_run(self):
        super(Antiship, self).step()
        # print("=================================================态势运行中")
        # terminal = False
        # return terminal

    def do_action(self, actions):
        """用于执行动作，并在环境中得到反馈"""
        step_use_mount = [0, 0, 0]                # 记录每种弹药使用数量
        #actions_int = [int(a) for a in actions]   # 将动作转换为整数

        self.prev_health = self.get_health()
        self.prev_mount = self.get_weapon_num()

        unit = self.blue_ship[0]["unit"]

        for action in [actions]:
            if action > 0:
                weapon = math.floor((action - 1) / self.n_enemies)
                target = action - weapon * self.n_enemies - 1
                loss_value = 0.2
                #if target < self.n_red_missile:                                             #         目标为拦截红弹
                    #target_guid = self.red_missile[target]["guid"]
                #else:                                                                       #         目标为打击红船
                target_guid = self.red_ship[0]["guid"]

                response = unit.allocate_weapon_to_target(target_guid, self.weapon_guid[weapon], 1)
                print(target_guid)
                if response == 'lua执行成功':
                    step_use_mount[weapon] += 1
                    #self.red_missile[target]["expect_value"] -= loss_value
                else:
                    error_ship = unit.strName

        self.scenario = super(Antiship, self).step()


        # for k, v in self.scenario.situation.trgunitdmgd_dic.ietms():                  #      trgunitdmgd_dic 毁伤情况

        #  计算拦截碰撞距离
        for k, v in self.scenario.situation.wpnimpact_dic.items():
            impact_lon = v.dLongitude
            impact_lat = v.dLatitude
            impact_dist = get_two_point_distance(self.center_lon, self.center_lat, impact_lon, impact_lat)
            self.all_impact_dist += impact_dist

        self.step += 1

        # 执行完这轮动作后，返回这步所有船使用各种弹药的综述，返回经过这步之后，所有船各种弹药的剩余量，返回是否拦截成功
        return step_use_mount

    def reward_battle(self, mount_use_ep):                                           # mount_use_ep
        # 初始化奖励列表和累计使用武器数量列表
        r = []
        mount_use_sum = [0, 0, 0]

        # 定义静态奖励的武器奖励列表
        weapon_reward = [-0.0006, -0.001, -0.0004]     # 值还没有改

        # 静态奖励，使用奖励列表 reward
        for mount_use_step in mount_use_ep:                                        # mount_use_ep
            step_reward = 0
            for weapon_id in range(len(weapon_reward)):
                step_reward += mount_use_step[weapon_id] * weapon_reward[weapon_id]
                mount_use_sum[weapon_id] += mount_use_step[weapon_id]
            r.append(step_reward)                                                                     # step_reward 蓝方使用弹的 单步奖励

        # 计算拦截奖励列表、拦截总和、终止标志、拦截列表、蓝方导弹状态
        #intercept_reward_list, intercept_sum, terminal, intercept_list, blue_missile = \
            #self.compute_intercept_reward_list_1()

        # 将奖励列表和拦截奖励列表相加
        # r = [[x + y] for x, y in zip(r, intercept_reward_list)]

        # 计算整个 episode 的总奖励
        episode_reward = sum(np.array(r).squeeze(1))

        # 计算平均冲击距离
        # average_impact_dist = self.all_impact_dist / 100000

        # 返回奖励列表、episode 总奖励、拦截总和、终止标志、累计使用武器数量列表、
        # 拦截列表、平均冲击距离、蓝方导弹状态
        return episode_reward
        #return r, episode_reward, intercept_sum, terminal, mount_use_sum, \
               #intercept_list, average_impact_dist, blue_missile

    def health_reward(self):
        blue_health = self.get_health()
        red_health = self.get_red_health()
        r = (1-red_health)/0.5*2-(1-blue_health)/0.5*3
        return r

    def mount_remain(self):
        # 初始化剩余武器数量列表
        step_remain_mount = [0, 0, 0]

        # 获取船只的生命值
        al_unit_health = self.get_health()  #

        # 更新当前生命值列表
        self.now_health[0] = al_unit_health

        # 如果船只的生命值为0，表示船只刚刚死亡
        if al_unit_health == 0:
            self.agent_is_dead[0] = 1  # 将船只标记为死亡
        else:
            # 获取船只的武器数量
            al_unit_mount = self.get_weapon_num()  # 假设蓝方阵营只有一个船只，索引为0

            # 统计剩余武器数量
            step_remain_mount[0] += al_unit_mount[0]
            step_remain_mount[1] += al_unit_mount[1]
            step_remain_mount[2] += al_unit_mount[2]

        # 返回剩余武器数量列表
        return step_remain_mount

    def time_to_step(self, time):
        """时间转换为步数"""
        hour = time[:2]
        # 30s一步的决策步长
        if hour[0] == "0":
            hour = hour[1]
        hour = int(hour)
        minute = time[-2:]
        if minute[0] == "0":
            minute = minute[1]
        minute = int(minute)
        # print("小时为：", hour, "分钟为", minute)
        time_interval = 30
        if time_interval == 15:
            if 17 > hour >= 13:
                time_step = (hour - 13) * 4
                if minute < 15:
                    return time_step  # 0
                elif 15 <= minute < 30:
                    return time_step + 1
                elif 30 <= minute < 45:
                    return time_step + 2
                else:
                    return time_step + 3
            elif hour >= 17:
                return self.step_limit - 1
        if time_interval == 30:
            if hour >= 13:
                time_step = (hour - 13) * 2
                if minute < 30:
                    return time_step
                else:
                    return time_step + 1

    def compute_intercept_reward_list_1(self):      # 静态拦截奖励
        # 设置Mozi日志文件存储路径
        path = "D:/Mozi/MoziServer/bin/Logs"

        # 获取路径下所有文件，并按修改时间排序
        lists = os.listdir(path)
        lists.sort(key=lambda fn: os.path.getmtime(path + "/" + fn))

        # 获取最新的日志文件路径和最旧的日志文件路径
        file_newest_path = os.path.join(path, lists[-1])  # 最新的
        wasted_file_path = os.path.join(path, lists[0])

        # 删除没有用的lua执行文件
        os.remove(wasted_file_path)

        # 打开最新的日志文件
        f = open(file_newest_path, encoding='utf-8')

        # 初始化拦截奖励列表、拦截计数列表、蓝方导弹信息字典
        intercept_reward_list = [0] * self.step_limit
        intercept_list = [0] * self.step_limit
        blue_missile = {}
        for key in range(self.step_limit):
            blue_missile[key] = []

        # 遍历日志文件的每一行
        for line in f:
            if '蓝方: 发射单元' in line:
                # 提取时间并转换为步数
                time = line[13:18]
                step = self.time_to_step(time)
                if step >= self.step_limit:
                    step = self.step_limit - 1

                # 提取蓝方导弹ID
                blue_missile_id_1 = line[-5:-3]

                # 将导弹ID添加到相应步数的列表中
                for key, value in blue_missile.items():
                    if key == step:
                        value.append(blue_missile_id_1)

            if '蓝方: 武器' in line and '爬升率' in line and '飞机' not in line:
                # 提取时间并转换为步数
                time = line[13:18]
                step = self.time_to_step(time)
                if step >= self.step_limit:
                    step = self.step_limit - 1

                # 更新拦截奖励列表和拦截计数列表
                intercept_reward_list[step] += 0.1
                intercept_list[step] += 1

        # 计算拦截奖励总和
        intercept_reward_sum = sum(intercept_reward_list)
        intercept_sum = round(intercept_reward_sum * 10)

        # 判断是否满足终止条件
        terminal = False
        if intercept_sum == self.n_enemies:
            terminal = True

        # 关闭文件并删除最新的日志文件
        f.close()
        os.remove(file_newest_path)

        # 返回拦截奖励列表、拦截总和、终止标志、拦截计数列表、蓝方导弹信息字典
        return intercept_reward_list, intercept_sum, terminal, intercept_list, blue_missile












