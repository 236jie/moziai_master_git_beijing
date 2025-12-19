
import unittest
from mozi_ai_sdk.test.env.env import Environment
from mozi_ai_sdk.test.env import etc
import os


class TestFramework(unittest.TestCase):

    def setUp(self):
        print("--------------- CASE START ----------------------------")

        os.environ['MOZIPATH'] = etc.MOZI_PATH

        # self.env = Environment(etc.SERVER_IP, etc.SERVER_PORT, etc.SERVER_PLAT,
        #                        etc.SCENARIO_NAME_SIMPLE_NETWORK, etc.SIMULATE_COMPRESSION,
        #                        etc.DURATION_INTERVAL, etc.SYNCHRONOUS, etc.app_mode)
        self.env = Environment('192.168.1.44', etc.SERVER_PORT, duration_interval=etc.DURATION_INTERVAL, app_mode=3,
                          agent_key_event_file=None, request_id='红方')
        # self.env = Environment(etc.SERVER_IP, etc.SERVER_PORT, duration_interval=etc.DURATION_INTERVAL, app_mode=3,
        #                   agent_key_event_file=None, request_id='红方')

        self.env.start()
        self.scenario = self.env.reset()
        self.red_side = self.scenario.get_side_by_name("红方")
        self.blue_side = self.scenario.get_side_by_name("蓝方")

        self.radar_1_guid = '4062dd33-f173-4013-9a7b-41eacdf0c042'
        self.radar_2_guid = 'e4a21970-280b-4f83-b7d2-ebfea88f8c65'
        self.radar_3_guid = '789c4c05-ec60-4e16-a4b2-879d3958fa71'
        self.radar_4_guid = '17000c06-be76-4a66-988e-c76a4294b3e8'
        self.center_guid = 'c8659905-54e4-4c59-b508-4e853918ce2f'

    def tearDown(self):
        print("--------------- CASE END ----------------------------")

    def test_add_network(self):
        """给两个单元添加网络"""
        # 想定精细度，选择简单通信网络
        self.scenario.set_fineness(sim_plex_network='true')
        self.red_side.add_network(self.radar_1_guid, self.center_guid, 0.7, 1, 4)
        for i in range(10):
            self.env.step()

        s = self.red_side.get_network_contact(self.center_guid, 6)
        pass


    def test_remove_network(self):
        self.scenario.set_fineness(sim_plex_network='true')
        self.red_side.add_network(self.radar_1_guid, self.center_guid, 0.7, 1, 4)
        self.red_side.remove_network(self.radar_1_guid, self.center_guid)
        for i in range(10):
            self.env.step()

        s = self.red_side.get_network_contact(self.center_guid, 2)
        pass


    def test_set_network(self):
        self.scenario.set_fineness(sim_plex_network='true')
        self.red_side.add_network(self.radar_1_guid, self.center_guid, 0.7, 1, 4)
        self.red_side.set_network(self.radar_1_guid, self.center_guid, 0.7, 600, 1200)
        for i in range(10):
            self.env.step()

        s = self.red_side.get_network_contact(self.center_guid, 2)
        pass

    def test_set_network_core(self):
        self.scenario.set_fineness(sim_plex_network='true')
        self.red_side.add_network(self.radar_1_guid, self.center_guid, 1.0, 1, 4)
        self.red_side.add_network(self.radar_4_guid, self.center_guid, 1.0, 1, 5)
        self.red_side.set_network_core(self.center_guid, air_cycle=60,  ship_cycle=300, sub_cycle=7200, fac_cycle=60)
        self.env.step()
        s1 = self.red_side.get_network_contact(self.radar_4_guid, 4)
        for i in range(15):
            self.env.step()
        s2 = self.red_side.get_network_contact(self.radar_4_guid, 2)
        s3 = self.red_side.get_network_contact(self.radar_4_guid, 4)
        s4 = self.red_side.get_network_contact(self.radar_4_guid, 6)
        self.env.step()
        s7 = self.red_side.get_network_contact(self.radar_4_guid, 2)
        self.env.step()
        s8 = self.red_side.get_network_contact(self.radar_4_guid, 2)
        self.env.step()
        s9 = self.red_side.get_network_contact(self.radar_4_guid, 2)
        self.env.step()
        s10 = self.red_side.get_network_contact(self.radar_4_guid, 2)
        self.env.step()
        s11 = self.red_side.get_network_contact(self.radar_4_guid, 2)
        pass

    def test_remove_network_core(self):
        self.scenario.set_fineness(sim_plex_network='true')
        self.red_side.add_network(self.radar_1_guid, self.center_guid, 0.7, 1, 4)
        self.red_side.add_network(self.radar_4_guid, self.center_guid, 0.7, 1, 5)
        self.red_side.set_network_core(self.center_guid, air_cycle=60,  ship_cycle=300, sub_cycle=7200, fac_cycle=60)
        self.red_side.remove_network(self.center_guid)
        self.env.step()
        s1 = self.red_side.get_network_contact(self.radar_4_guid, 2)
        for i in range(60):
            self.env.step()
        s2 = self.red_side.get_network_contact(self.radar_4_guid, 2)
        pass


    # 两个雷达同时探测目标，探测的信息不同的情况