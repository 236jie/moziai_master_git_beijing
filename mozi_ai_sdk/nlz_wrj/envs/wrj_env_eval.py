# 时间 ： 2020/8/29 20:34
# 作者 ： Dixit
# 文件 ： tasks.py
# 项目 ： moziAIBT2
# 版权 ： 北京华戍防务技术有限公司

from mozi_ai_sdk.btmodel.bt import utils
import re
import random
import sys
import itertools
import uuid
from collections import namedtuple
import datetime
import numpy as np
from itertools import chain
import pdb
import time
import docker
from mozi_simu_sdk.mssnpatrol import CPatrolMission
from mozi_simu_sdk.mssnstrike import CStrikeMission
from mozi_ai_sdk.nlz_wrj.envs.env import Environment
from mozi_ai_sdk.nlz_wrj.envs import etc

import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Discrete, Box, Dict
Function = namedtuple('Function', ['type', 'function', 'is_valid'])

def restart_container(port):
    # docker init
    client = docker.from_env()
    try:
        container = client.containers.get('pydockertest_%s' % port)
        container.stop()
        container.remove()
    except Exception:
        print('fail get or stop or remove pydockertest_%s container!' % port)
        sys.exit(1)
    try:
        while True:
            while True:
                container = client.containers.create('mozi_innet_v16',
                                                     command='/bin/bash',
                                                     name='pydockertest_%s' % str(port),
                                                     detach=True,
                                                     tty=True,
                                                     ports={'6060': port},
                                                     user='root')
                container.start()
                out = container.exec_run(cmd='sh -c "service mysql start && echo success"',
                                         tty=True,
                                         user='root',
                                         detach=False)
                print(out.output)
                if 'started' in out.output.decode('utf-8'):
                    break
                print('pydockertest_%s mysql fail to start!' % str(port))
                container.stop()
                container.remove()
            print('pydockertest_%s mysql is started!' % str(port))
            container.exec_run(cmd='sh -c "mono /home/LinuxServer/bin/LinuxServer.exe --AiPort 6060"',
                               tty=True,
                               user='root',
                               detach=True)
            time.sleep(2)
            out2 = container.exec_run(cmd='sh -c "pgrep mono"',
                                      tty=True,
                                      user='root',
                                      detach=False)
            if out2.output != b'':
                print('pydockertest_%s mozi is started!' % str(port))
                break
            print('pydockertest_%s mozi fail to start!' % str(port))
            container.stop()
            container.remove()
        return port
    except Exception:
        print('fail create mozi docker!')
        sys.exit(1)

class WRJ(MultiAgentEnv):
    def __init__(self, env_config):
        self.env_config = env_config
        self.reset_nums = 0
        if env_config['mode'] == 'train':
            while True:
                    try:
                        env = Environment(etc.SERVER_IP,
                                          etc.SERVER_PORT,
                                          'linux',
                                          etc.SCENARIO_NAME,
                                          etc.SIMULATE_COMPRESSION,
                                          etc.DURATION_INTERVAL,
                                          etc.SYNCHRONOUS)
                        # by dixit
                        if env_config['avail_port']:
                            self.avail_port_list = env_config['avail_port']
                        else:
                            raise Exception('no avail port!')
                        # self.self.reset_nums = 0
                        self.ip = '127.0.0.1'
                        self.ip_port = f'{self.ip}:{self.avail_port_list[0]}'
                        print(self.ip_port)
                        env.start(self.ip, self.avail_port_list[0])
                        break
                    except Exception:
                        continue
        elif env_config['mode'] == 'eval':
            env = Environment(etc.SERVER_IP,
                              etc.SERVER_PORT, 
                              'windows',
                              etc.EVAL_SCENARIO_NAME,
                              etc.SIMULATE_COMPRESSION,
                              etc.DURATION_INTERVAL,
                              etc.SYNCHRONOUS)
            # by dixit
            port = '6060'
            ip = '127.0.0.1'
            env.start(ip, port)
            self.ip_port = f'{ip}:{port}'
        
        scenario = env.reset('红方')
        self.scenario = scenario
        self.time = self.scenario.m_Duration.split('@')     # 想定总持续时间
        self.m_StartTime = self.scenario.m_StartTime    # 想定开始时间
        self.m_Time = self.scenario.m_Time  # 想定当前时间
        self._env = env
        self.sideName = '红方'
        self.h_sideName = '蓝方'
        self.side = self.scenario.get_side_by_name(self.sideName)
        self.h_side = self.scenario.get_side_by_name(self.h_sideName)
        self.reward = 0.0
        self.action_reward = 0.0
        self.last_reward = 0.0
        self.score = 0.0
        self.miss_1 = list(itertools.product([x for x in range(6)], [y for y in range(5)]))
        self.miss_2 = list(itertools.product([x for x in range(3)], [z for z in range(3)]))
        self.miss = self.miss_1 + self.miss_2
        self.unit_record = {'EL/M-2106': False, 'EL/M-2084': False, '反辐射巡飞弹0': False,
                            '反辐射巡飞弹1': False, '反辐射巡飞弹2': False, '反辐射巡飞弹0加分': False,
                            '反辐射巡飞弹1加分': False, '反辐射巡飞弹2加分': False, 'ddc': 0}  # 参与计分的单元
        n_dims = 2*8 + 3*8 + 3 + 39
        self.action_space = Discrete(6*5 + 3*3)
        self.ac_feat = [0.0 for _ in range(39)]
        self.observation_space = Dict({
            "obs": Box(float('-inf'), float('inf'), shape=(n_dims,))
        })
        self.assigned_ffs = []
        self.assigned_hjp = []
        self.assigned_ddc = []
        self.ddc_recorded = []
        self.total_ddc = [v.strGuid for _, v in self.h_side.facilities.items() if 'SPYDER' in v.strName]

    def _update(self, scenario):
        self.scenario = scenario
        self.side = self.scenario.get_side_by_name(self.sideName)
        self.h_side = self.scenario.get_side_by_name(self.h_sideName)
        self.time = self.scenario.m_Duration.split('@')
        self.m_StartTime = self.scenario.m_StartTime  # 想定开始时间
        self.m_Time = self.scenario.m_Time  # 想定当前时间

    def reward_schema(self):
        radar = [v.strName for k, v in self.h_side.facilities.items() if 'EL/M' in v.strName]
        ddc = [v.strGuid for _, v in self.h_side.facilities.items() if 'SPYDER' in v.strName]
        if ('EL/M-2106 ATAR三维坐标监视雷达' not in radar) and (self.unit_record['EL/M-2106'] is False):
            self.reward += 0.5
            self.score += 0.5
            self.unit_record['EL/M-2106'] = True
        if ('EL/M-2084 MMR型雷达' not in radar) and (self.unit_record['EL/M-2084'] is False):
            self.reward += 0.5
            self.score += 0.5
            self.unit_record['EL/M-2084'] = True
        for guid in self.total_ddc:
            if (guid not in ddc) and (guid not in self.ddc_recorded):
                self.reward += 0.1
                self.score += 0.1
                self.ddc_recorded.append(guid)
        xfd = [v.strName for k, v in self.side.aircrafts.items() if '反辐射巡飞弹' in v.strName]
        for index in range(3):
            if ('反辐射巡飞弹%d' % index not in xfd) and (self.unit_record['反辐射巡飞弹%d' % index] is False):
                self.reward -= 0.5
                self.score -= 0.5
                self.unit_record['反辐射巡飞弹%d' % index] = True
        self.ffs_zh()

    def execute_action(self, ac):
        ac_index = self.miss[ac['agent_0']]
        if ac['agent_0'] <= 29:
            dd_target_name = ['#129', '#130', '#131', '#132', '#133', '#134']
            hjp_unit_name = ['无人值守火箭炮 #1', '无人值守火箭炮 #2', '无人值守火箭炮 #3', '无人值守火箭炮 #4', '无人值守火箭炮 #5']
            action = {'target_name': dd_target_name[ac_index[0]], 'mission_unit_name': hjp_unit_name[ac_index[1]]}
            # 火箭炮打击地空导弹发射车
            target_name = action['target_name']  # ['#129', '#130', '#131', '#132', '#133', '#134']
            missionName = 'hjp_attack_ddc' + target_name
            # ['无人值守火箭炮 #1', '无人值守火箭炮 #2', '无人值守火箭炮 #3', '无人值守火箭炮 #4', '无人值守火箭炮 #5']
            mission_unit_name = action['mission_unit_name']
            target = {k: v for k, v in self.side.contacts.items() if target_name in v.strName}
            if target:
                pass
            else:
                return
            if target_name in self.assigned_ddc:
                return
            if mission_unit_name in self.assigned_hjp:
                return
            missionUnit = {k: v for k, v in self.side.facilities.items() if mission_unit_name in v.strName}
            self.add_hjp_dj(missionName, target, missionUnit)
            self.assigned_ddc.append(target_name)
            self.assigned_hjp.append(mission_unit_name)
            
            # reward schema
            if target_name != '#133':
                self.reward += 0.2
                self.score += 0.2
            else:
                self.reward -= 1.0
                self.score -= 1.0
            if len(self.assigned_ffs) == 0:
                self.reward += 0.2
                self.score += 0.2
            else:
                self.reward -= 0.2
                self.score -= 0.2
        else:
            ffs_unit_index = [0, 1, 2]
            # hx_index = [i for i in range(31)]
            # ffs_target_name = ['EL/M-2106', 'EL/M-2084']
            action = {'ffs_index': ffs_unit_index[ac_index[0]]}
            # 反辐射巡飞弹打击地空导弹发射车
            # 反辐射巡飞弹[0,1,2],沿航线[0-30]，打击['EL/M-2106', 'EL/M-2084']
            ffs_index = action['ffs_index']  # [0,1,2]
            if ac_index[-1] != 0:
                return
            if ffs_index in self.assigned_ffs:
                return
            # hx_index = action['hx_index']  # [0-30]
            self.ffs_dj(ffs_index)
            self.assigned_ffs.append(ffs_index)

            # reward schema
            if ('#129' in self.assigned_ddc) and ('#130' in self.assigned_ddc) and ('#131' in self.assigned_ddc) and ('#132' in self.assigned_ddc) and ('#134' in self.assigned_ddc):
                self.reward += 1.0
                self.score += 1.0
            if ('#129' not in self.assigned_ddc) or ('#130' not in self.assigned_ddc) or ('#131' not in self.assigned_ddc) or ('#132' not in self.assigned_ddc) or ('#134' not in self.assigned_ddc):
                self.reward -= 0.5
                self.score -= 0.5
            # time.sleep(5)

    def step(self, action):
        self.execute_action(action)
        scenario = self._env.step()     # 墨子环境step
        self._update(scenario)
        self.reward_schema()
        self.ac_feat[action['agent_0']] = 1.0
        feats = self.generate_features()
        self.action_reward = self.reward - self.last_reward
        self.last_reward = self.reward
        print(self.ip_port, ' ', f'action_reward:{self.action_reward}', ' ', f'score:{self.score}')
        done = self._is_done()
        obs = {'agent_0': {"obs": feats}}
        reward = {"agent_0": self.action_reward}
        return obs, reward, {"__all__": done, "agent_0": done}, {}

    def reset(self):
        self.reset_nums += 1
        if self.env_config['mode'] == 'train':
            if self.reset_nums % 5 == 0:
                port = self.avail_port_list[0]
                while True:
                    try:
                        restart_container(port)
                        env = Environment(etc.SERVER_IP,
                                          etc.SERVER_PORT,
                                          'linux',
                                          etc.SCENARIO_NAME,
                                          etc.SIMULATE_COMPRESSION,
                                          etc.DURATION_INTERVAL,
                                          etc.SYNCHRONOUS)
                        env.start(self.ip, port)
                        break
                    except Exception:
                        continue
                self._env = env
                scenario = self._env.reset(self.sideName)
            else:
                scenario = self._env.reset(self.sideName)
        else:
            scenario = self._env.reset(self.sideName)
        self.reward = 0.0
        self.action_reward = 0.0
        self.last_reward = 0.0
        self.score = 0.0
        self.unit_record = {'EL/M-2106': False, 'EL/M-2084': False, '反辐射巡飞弹0': False,
                                    '反辐射巡飞弹1': False, '反辐射巡飞弹2': False, '反辐射巡飞弹0加分': False,
                                    '反辐射巡飞弹1加分': False, '反辐射巡飞弹2加分': False, 'ddc': 0}  # 参与计分的单元
        self.assigned_ffs = []
        self.assigned_hjp = []
        self.assigned_ddc = []
        self.ddc_recorded = []
        self.total_ddc = [v.strGuid for _, v in self.h_side.facilities.items() if 'SPYDER' in v.strName]
        self.ac_feat = [0.0 for _ in range(39)]
        self._update(scenario)
        self.add_ffs_uav()
        # self.add_hw_uav()
        # for index in range(31):
        #     self.generate_plan_way(index)
        scenario = self._env.step()
        self._update(scenario)
        feats = self.generate_features()
        obs = {'agent_0': {"obs": feats}}
        return obs

    def generate_features(self):
        feats = []
        h_feats = [0.0, 0.0]
        targets_name = ['#129', '#130', '#131', '#132', '#133', '#134', 'EL/M-2106', 'EL/M-2084']
        for target_name in targets_name:
            for k, v in self.side.contacts.items():
                if target_name in v.strName:
                    h_feats[0] = v.dLongitude / 90
                    h_feats[1] = v.dLatitude / 30
                    break
            feats.extend(h_feats)
            h_feats[0] = 0.0
            h_feats[1] = 0.0
        red_feats = [0.0, 0.0, 0.0]
        units_name = ['反辐射巡飞弹0', '反辐射巡飞弹1', '反辐射巡飞弹2',
                      '无人值守火箭炮 #1', '无人值守火箭炮 #2', '无人值守火箭炮 #3', '无人值守火箭炮 #4', '无人值守火箭炮 #5']
        for unit_name in units_name:
            for k, v in self.side.facilities.items():
                if unit_name in v.strName:
                    red_feats[0] = v.dLongitude / 90
                    red_feats[1] = v.dLatitude / 30
                    red_feats[2] = v.fCurrentHeading / 360
                    break
            feats.extend(red_feats)
            red_feats[0] = 0.0
            red_feats[1] = 0.0
            red_feats[2] = 0.0
        time_delta = self.m_Time - self.m_StartTime
        feats.append(time_delta / 3600)
        feats.append(time_delta / 7200)
        feats.append(time_delta / 14400)
        feats.extend(self.ac_feat)

        return feats

    def _is_done(self):
        # pdb.set_trace()
        if len(self.time) == 1:
            return False
        if self.time[0] == '' or self.time[1] == '' or self.time[2] == '':
            return False
        duration = int(self.time[0]) * 86400 + int(self.time[1]) * 3600 + int(self.time[2]) * 60
        if self.m_StartTime + duration <= self.m_Time + 30:
            return True
        else:
            pass

        # 对战平台
        # response_dic = self.scenario.get_responses()
        # for _, v in response_dic.items():
        #     if v.Type == 'EndOfDeduction':
        #         print('打印出标记：EndOfDeduction')
        #         return True
        # return False

    def generate_plan_way(self, index):
        '''
        生成预设航线
        :param index:
        :return:
        '''
        self.side.add_plan_way(0, 'ffs_dj_hx_%d' % index)
        for k, v in self.side.referencepnts.items():
            if 'RP-454' in v.strName:
                self.side.add_plan_way_point('ffs_dj_hx_%d' % index, v.dLongitude, v.dLatitude)
        for k, v in self.side.referencepnts.items():
            if 'plan-%d' % index == v.strName:
                self.side.add_plan_way_point('ffs_dj_hx_%d' % index, v.dLongitude, v.dLatitude)
        for k, v in self.side.referencepnts.items():
            if 'RP-457' in v.strName:
                self.side.add_plan_way_point('ffs_dj_hx_%d' % index, v.dLongitude, v.dLatitude)
        for k, v in self.side.referencepnts.items():
            if 'RP-458' in v.strName:
                self.side.add_plan_way_point('ffs_dj_hx_%d' % index, v.dLongitude, v.dLatitude)

    def ffs_dj(self, ffs_index):
        '''
        反辐射打击任务
        :param ffs_index: 0, 1, 2
        :param target_name: 'EL/M-2106', 'EL/M-2084'
        :param index: 0-30
        :return:
        '''
        for k, v in self.side.aircrafts.items():
            # 为反辐射巡飞弹添加预设航线
            if '反辐射巡飞弹%d' % ffs_index in v.strName and len(v.m_MultipleMissionGUIDs) == 0:
                missionName = '反辐射巡飞弹%d巡逻任务' % ffs_index
                missionUnit = {k: v}
                self.add_xl_mission(missionName, missionUnit)
            # 反辐射巡飞弹武器打完自毁
            if '反辐射巡飞弹' in v.strName:
                weapon = v.m_UnitWeapons.split('x')
                # print(weapon)
                if weapon[0] == '':
                    # cmd = "ScenEdit_DeleteUnit({name = '%s'})" % v.strName
                    # self.scenario.mozi_server.send_and_recv(cmd)
                    if self.unit_record[v.strName + '加分'] is False:
                        self.reward += 0.5
                        self.score += 0.5
                        self.unit_record[v.strName + '加分'] = True

    def ffs_zh(self):
        for k, v in self.side.aircrafts.items():
            # 反辐射巡飞弹武器打完自毁
            if '反辐射巡飞弹' in v.strName:
                weapon = v.m_UnitWeapons.split('x')
                # print(weapon)
                if weapon[0] == '':
                    cmd = "ScenEdit_DeleteUnit({name = '%s'})" % v.strName
                    self.scenario.mozi_server.send_and_recv(cmd)
                    if self.unit_record[v.strName + '加分'] is False:
                        self.reward += 0.5
                        self.score += 0.5
                        self.unit_record[v.strName + '加分'] = True

    def add_ffs_uav(self):
        '''
        添加反辐射无人机，用于攻击雷达
        UAVUnit = ScenEdit_AddUnit({side=sideName,unitname="反辐射巡飞弹 #0"..index,type="Air",dbid=1004964,latitude=unit.latitude,longitude=unit.longitude,loadoutid=1025689,altitude=high})
        :return:
        '''
        # for k, v in self.side.facilities.items():
        #     if k == 'ed523852-66e9-48a7-871d-98fcfe96367b':
        #         latitude = v.dLatitude
        #         longitude = v.dLongitude
        num = 3
        for index in range(num):
            cmd = "ScenEdit_AddUnit({side = \'%s\', " \
                  "unitname = \'反辐射巡飞弹%d\', type = \'Air\', dbid = 1004964, " \
                  "latitude = 27.70302, longitude = 89.134928, " \
                  "loadoutid = 1025689, altitude = 300})" % (self.sideName, index)
            self.scenario.mozi_server.send_and_recv(cmd)

    def add_hw_uav(self):
        '''
        添加红外无人机，用于攻击移动车辆
        UAVUnit =ScenEdit_AddUnit({side=sideName,unitname="攻击移动车辆巡飞弹 #0"..index,type="Air",dbid=1004965,latitude=unit.latitude,longitude=unit.longitude,loadoutid=1025690,altitude=high})
        :return:
        '''
        num = 3
        for index in range(num):
            cmd = "ScenEdit_AddUnit({side = \'%s\', " \
                  "unitname = \'红外巡飞弹%d\', type = \'Air\', dbid = 1004965, " \
                  "latitude = 27.690823, longitude = 89.137051, " \
                  "loadoutid = 1025690, altitude = 300})" % (self.sideName, index)
            self.scenario.mozi_server.send_and_recv(cmd)

    def add_hjp_dj(self, missionName, target, missionUnit):
        '''
        添加火箭炮打击地空导弹发射车的任务
        :param missionName:
        :param target: 地空导弹#129, #130, #131, #132, #133, #134
        :param missionUnit:无人值守火箭炮 #1, 无人值守火箭炮 #2, 无人值守火箭炮 #3
        :return:
        '''
        side = self.side
        strikemssn = [v for _, v in side.strikemssns.items() if v.strName == missionName]
        if len(strikemssn) != 0:
            return False

        scen_time = '07/29/2020 14:38:52'
        mission_time = datetime.datetime.strptime(scen_time, '%m/%d/%Y %H:%M:%S') + datetime.timedelta(
            minutes=1)
        side.add_mission_strike(missionName, 1)
        AntiSurface = CStrikeMission('T+1_mode', self.scenario.mozi_server, self.scenario.situation)
        AntiSurface.m_Side = self.sideName
        AntiSurface.strName = missionName
        taskParam = {'missionName': missionName, 'missionType': '对陆打击', 'flightSize': 1, 'checkFlightSize': True,
                     'startTime': '%s' % str(mission_time),
                     'endTime': '08/09/2020 12:00:00', 'isActive': 'true', 'missionUnit': missionUnit,
                     'targets': target}
        AntiSurface.set_preplan(True)  # 仅考虑计划目标
        self._SetTaskParam(AntiSurface, taskParam)
        print(f'created hjp{missionName}')

    def add_xl_mission(self, missionName, missionUnit):
        '''
        添加反地面站巡逻任务
        :return:
        '''
        side = self.side
        zone = ['RP-450', 'RP-451', 'RP-452', 'RP-453']
        patrolmssn = [v for _, v in side.patrolmssns.items() if v.strName == missionName]
        if len(patrolmssn) != 0:
            return False
        scen_time = '07/29/2020 14:38:52'
        mission_time = datetime.datetime.strptime(scen_time, '%m/%d/%Y %H:%M:%S') + datetime.timedelta(
            minutes=1)
        side.add_mission_patrol(missionName, 2, zone)  # 空战巡逻
        DefensiveAirMiss = CPatrolMission('T+1_mode', self.scenario.mozi_server, self.scenario.situation)
        DefensiveAirMiss.strName = missionName
        # cmd = f"Hs_AddPlanWayToMission('{missionName}', 0, '{hx_name}')"
        # self.scenario.mozi_server.send_and_recv(cmd)
        taskParam = {'missionName': missionName, 'missionType': '空战巡逻', 'flightSize': 1, 'checkFlightSize': True,
                     'oneThirdRule': True, 'chechOpa': False, 'checkWwr': False, 'startTime': '%s' % str(mission_time),
                     'endTime': '08/09/2020 12:00:00', 'isActive': 'true', 'missionUnit': missionUnit}
        self._SetTaskParam(DefensiveAirMiss, taskParam)
        print(f'missionName:{missionName} ', f'len(missionUnit):{len(missionUnit)}')


    def side_info(self):
        '''
        我方单元的信息：经纬度、航向、航速、高度、载弹量等，用于构造状态空间
        :return:
        '''
        pass

    def contacts_info(self):
        '''
        敌方单元的信息：经纬度、航向、航速、高度、载弹量等，用于构造状态空间
        :return:
        '''
        pass

    # 修改任务参数
    def _SetTaskParam(self, mission, kwargs):
        # kwargs = {'missionName': miss1, 'missionType': '空战巡逻', 'flightSize': 2, 'checkFlightSize': True, 'oneThirdRule': True,
        #           'chechOpa': False, 'checkWwr': True, 'startTime': '08/09/2020 00:00:00',
        #           'endTime': '08/09/2020 12:00:00', 'isActive': 'true', 'missionUnit': , 'targets': }
        kwargs_keys = kwargs.keys()
        # 设置编队规模
        if 'flightSize' in kwargs_keys:
            # mission.set_flight_size(self.sideName, kwargs['missionName'], kwargs['flightSize'])
            mission.set_flight_size(kwargs['flightSize'])
        # 检查编队规模
        if 'checkFlightSize' in kwargs_keys:
            # mission.set_flight_size_check(self.sideName, kwargs['missionName'], True)
            mission.set_flight_size_check( True)
        # 设置1/3规则
        if 'oneThirdRule' in kwargs_keys:
            mission.set_one_third_rule(kwargs['oneThirdRule'])
        # 是否对巡逻区外的探测目标进行分析
        if 'chechOpa' in kwargs_keys:
            mission.set_opa_check(kwargs['chechOpa'])
        # 是否对武器射程内探测目标进行分析
        if 'checkWwr' in kwargs_keys:
            mission.set_wwr_check(kwargs['checkWwr'])
        # 设置任务的开始和结束时间
        if 'startTime' in kwargs_keys:
            cmd_str = "ScenEdit_SetMission('" + self.sideName + "','" + kwargs['missionName'] + "',{starttime='" + kwargs['startTime'] + "'})"
            self.scenario.mozi_server.send_and_recv(cmd_str)
        if 'endTime' in kwargs_keys:
            cmd_str = "ScenEdit_SetMission('" + self.sideName + "','" + kwargs['missionName'] + "',{endtime='" + kwargs['endTime'] + "'})"
            self.scenario.mozi_server.send_and_recv(cmd_str)
        # 设置是否启动任务
        if 'isActive' in kwargs_keys:
            lua = "ScenEdit_SetMission('%s','%s',{isactive='%s'})" % (self.sideName, kwargs['missionName'], kwargs['isActive'])
            self.scenario.mozi_server.send_and_recv(lua)
        if 'missionUnit' in kwargs_keys:
            mission.assign_units(kwargs['missionUnit'])
        if 'targets' in kwargs_keys:
            # mission.assign_targets(kwargs['targets'])
            self.side.assign_target_to_mission(kwargs['targets'], mission.strName)

    # 修改任务条令、电磁管控
    def _SetTaskDoctrineAndEMC(self, doctrine, kwargs):
        # kwargs = {'emc_radar': 'Passive', 'evadeAuto': 'true', 'ignorePlottedCourse': 'yes', 'targetsEngaging': 'true',
        #           'ignoreEmcon': 'false', 'weaponControlAir': '0', 'weaponControlSurface': '0', 'fuelStateForAircraft': '0',
        #           'fuelStateForAirGroup': '3', 'weaponStateForAircraft': '2001', 'weaponStateForAirGroup': '3'}

        kwargs_keys = kwargs.keys()

        # 电磁管控
        # em_item: {str: 'Radar' - 雷达, 'Sonar' - 声呐, 'OECM' - 光电对抗}
        # status: {str: 'Passive' - 仅有被动设备工作, 'Active' - 另有主动设备工作
        if 'emc_radar' in kwargs_keys:
            doctrine.set_em_control_status(em_item='Radar', status=kwargs['emc_radar'])
        # 设置是否自动规避
        if 'evadeAuto' in kwargs_keys:
            doctrine.evade_automatically(kwargs['evadeAuto'])
        # 设置是否忽略计划航线
        if 'ignorePlottedCourse' in kwargs_keys:
            doctrine.ignore_plotted_course(kwargs['ignorePlottedCourse'])
        # 接战临机出现目标
        # opportunity_targets_engaging_status: {str: 'true' - 可与任何目标交战, 'false' - 只与任务相关目标交战}
        if 'targetsEngaging' in kwargs_keys:
            doctrine.set_opportunity_targets_engaging_status(kwargs['targetsEngaging'])
        # 受攻击时是否忽略电磁管控
        if 'ignoreEmcon' in kwargs_keys:
            doctrine.ignore_emcon_while_under_attack(kwargs['ignoreEmcon'])

        # 设置武器控制状态
        # domain: {str: 'weapon_control_status_subsurface' - 对潜,
        #               'weapon_control_status_surface' - 对面,
        #               'weapon_control_status_land' - 对陆,
        #               'weapon_control_status_air' - 对空}
        # fire_status: {str: '0' - 自由开火, '1' - 谨慎开火, '2' - 限制开火}
        if 'weaponControlAir' in kwargs_keys:
            doctrine.set_weapon_control_status(domain='weapon_control_status_air', fire_status=kwargs['weaponControlAir'])
        if 'weaponControlSurface' in kwargs_keys:
            doctrine.set_weapon_control_status(domain='weapon_control_status_surface', fire_status=kwargs['weaponControlSurface'])

        # 设置单架飞机返航的油料状态
        if 'fuelStateForAircraft' in kwargs_keys:
            doctrine.set_fuel_state_for_aircraft(kwargs['fuelStateForAircraft'])
        # 设置飞行编队返航的油料状态
        # fuel_state: {str:   'No'('0') - 无约束，编队不返航,
        #                     'YesLastUnit'('1') - 编队中所有飞机均因达到单机油料状态要返航时，编队才返航,
        #                     'YesFirstUnit'('2') - 编队中任意一架飞机达到单机油料状态要返航时，编队就返航,
        #                     'YesLeaveGroup'('3') - 编队中任意一架飞机达到单机油料状态要返航时，其可离队返航}
        if 'fuelStateForAirGroup' in kwargs_keys:
            doctrine.set_fuel_state_for_air_group(kwargs['fuelStateForAirGroup'])
        # 设置单架飞机的武器状态
        if 'weaponStateForAircraft' in kwargs_keys:
            doctrine.set_weapon_state_for_aircraft(kwargs['weaponStateForAircraft'])
        # 设置飞行编队的武器状态
        # weapon_state: {str: 'No'('0') - 无约束，编队不返航,
        #                     'YesLastUnit'('1') - 编队中所有飞机均因达到单机武器状态要返航时，编队才返航,
        #                     'YesFirstUnit'('2') - 编队中任意一架飞机达到单机武器状态要返航时，编队就返航,
        #                     'YesLeaveGroup'('3') - 编队中任意一架飞机达到单机武器状态要返航时，其可离队返航}
        if 'weaponStateForAirGroup' in kwargs_keys:
            doctrine.set_weapon_state_for_air_group(kwargs['weaponStateForAirGroup'])

