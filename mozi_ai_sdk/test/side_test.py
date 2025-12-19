from mozi_ai_sdk.test.utils.test_framework import TestFramework
from mozi_ai_sdk.test.utils import common


class TestSide(TestFramework):
    """测试推演方类"""

    def test_get_weapon_db_guids(self):
        """获取推演方所有武器的数据库guid"""
        weapon_db_guids = self.red_side.get_weapon_db_guids()
        self.env.step()
        self.assertTrue('hsfw-dataweapon-00000000001650' in weapon_db_guids)

    def test_get_weapon_infos(self):
        """获取推演方所有武器的名称及数据库guid"""
        weapon_infos = self.red_side.get_weapon_infos()
        self.env.step()
        self.assertTrue(['16x 89型鱼雷[G-12]', 'hsfw-dataweapon-00000000001650'] in weapon_infos)

    def test_get_groups(self):
        """获取本方编组"""
        groups = self.red_side.get_groups()
        count = 0
        for k, v in groups.items():
            count += 1
            self.assertTrue(v.strName in ['飞行编队 37', '飞行编队 59', '日本舰队'])
        self.assertEqual(count, 3)

    def test_get_submarines(self):
        """获取本方潜艇"""
        submarines = self.red_side.get_submarines()
        self.assertTrue(submarines)
        for k, v in submarines.items():
            self.assertTrue(v.strName in ['潜艇1'])

    def test_get_ships(self):
        """获取本方船"""
        ships = self.red_side.get_ships()
        self.assertTrue(ships)
        for k, v in ships.items():
            self.assertEqual(v.ClassName, 'CShip')

    def test_get_facilities(self):
        """获取本方地面单位"""
        facilities = self.red_side.get_facilities()
        self.assertTrue(facilities)
        for k, v in facilities.items():
            self.assertEqual(v.ClassName, 'CFacility')

    def test_get_aircrafts(self):
        """获取本方飞机"""
        aircrafts = self.red_side.get_aircrafts()
        self.assertTrue(aircrafts)
        for k, v in aircrafts.items():
            self.assertEqual(v.ClassName, 'CAircraft')

    def test_get_satellites(self):
        """获取本方卫星"""
        satellites = self.red_side.get_satellites()
        self.assertTrue(satellites)
        for k, v in satellites.items():
            self.assertEqual(v.ClassName, 'CSatellite')

    def test_get_weapons(self):
        """获取本方武器"""
        weapons = self.red_side.get_weapons()
        self.assertTrue(weapons)
        for k, v in weapons.items():
            self.assertEqual(v.ClassName, 'CWeapon')

    def test_get_unguided_weapons(self):
        """获取本方非制导武器"""
        area = ['RP-66', 'RP-67', 'RP-68', 'RP-69']
        self.red_side.deploy_mine('hsfw-dataweapon-00000000000634', 2, area)
        self.env.step()
        weapons_1 = self.red_side.get_unguided_weapons()
        count = 0
        for k, v in weapons_1.items():
            count += 1
            self.assertEqual(v.strDBGUID, 'hsfw-dataweapon-00000000000634')
        self.assertEqual(count, 2)

    def test_get_sideways(self):
        """获取预定义航路"""
        sideways = self.red_side.get_sideways()
        self.assertTrue(sideways)
        for k, v in sideways.items():
            self.assertEqual(v.ClassName, 'CSideWay')

    def test_get_contacts(self):
        """获取本方目标"""
        contacts = self.red_side.get_contacts()
        self.assertTrue(contacts)
        for k, v in contacts.items():
            self.assertEqual(v.ClassName, 'CContact')

    def test_get_logged_messages(self):
        """获取本方日志消息"""
        # TODO 返回值为空
        self.env.step()
        logged_messages = self.red_side.get_logged_messages()
        self.assertTrue(logged_messages)
        for k, v in logged_messages.items():
            self.assertEqual(v.ClassName, 'XXX')

    def test_get_patrol_missions(self):
        """获取巡逻任务"""
        mission = self.red_side.get_patrol_missions()
        self.assertTrue(mission)
        for k, v in mission.items():
            self.assertEqual(v.ClassName, 'CPatrolMission')

    def test_get_strike_missions(self):
        """获取打击任务"""
        mission = self.red_side.get_strike_missions()
        self.assertTrue(mission)
        for k, v in mission.items():
            self.assertEqual(v.ClassName, 'CStrikeMission')

    def test_get_support_missions(self):
        """获取支援任务"""
        mission = self.red_side.get_support_missions()
        self.assertTrue(mission)
        for k, v in mission.items():
            self.assertEqual(v.ClassName, 'CSupportMission')

    def test_get_cargo_missions(self):
        """获取运输任务"""
        mission = self.red_side.get_cargo_missions()
        self.assertTrue(mission)
        for k, v in mission.items():
            self.assertEqual(v.ClassName, 'CCargoMission')

    def test_get_ferry_missions(self):
        """获取转场任务"""
        mission = self.red_side.get_ferry_missions()
        self.assertTrue(mission)
        for k, v in mission.items():
            self.assertEqual(v.ClassName, 'CFerryMission')

    def test_get_mining_missions(self):
        """获取布雷任务"""
        # TODO 接口返回为空
        mission = self.red_side.get_mining_missions()
        self.assertTrue(mission)
        for k, v in mission.items():
            self.assertEqual(v.ClassName, 'CMiningMission')

    def test_get_mine_clearing_missions(self):
        """获取扫雷任务"""
        # TODO 接口返回为空
        mission = self.red_side.get_mine_clearing_missions()
        self.assertTrue(mission)
        for k, v in mission.items():
            self.assertEqual(v.ClassName, 'CMineClearingMission')

    def test_get_missions_by_name(self):
        """根据任务名称获取任务"""
        mission = self.red_side.get_missions_by_name('转场')
        self.assertTrue(mission)
        self.assertEqual(mission.strName, '转场')

    def test_get_reference_points(self):
        """获取参考点"""
        points = self.red_side.get_reference_points()
        self.assertTrue(points)
        for k, v in points.items():
            self.assertEqual(v.ClassName, 'CReferencePoint')

    def test_get_no_nav_zones(self):
        """获取禁航区"""
        zones = self.red_side.get_no_nav_zones()
        self.assertTrue(zones)
        for k, v in zones.items():
            self.assertEqual(v.ClassName, 'CNoNavZone')

    def test_set_reference_point(self):
        """更新参考点坐标"""
        self.red_side.set_reference_point('RP-31', 28, 130)
        self.env.step()
        points = self.red_side.get_reference_points()
        flag = False
        for k, v in points.items():
            if v.strName == 'RP-31':
                flag = True
                self.assertEqual(v.dLongitude, 130.0)
                self.assertEqual(v.dLatitude, 28.0)
        self.assertTrue(flag)

    def test_get_exclusion_zones(self):
        """获取封锁区"""
        zones = self.red_side.get_exclusion_zones()
        self.assertTrue(zones)
        for k, v in zones.items():
            self.assertEqual(v.ClassName, 'CExclusionZone')

    def test_get_score(self):
        """获取本方分数"""
        # 设置2021/7/19 9:23:00，红方得分100
        self.scenario.add_trigger_time('时间触发器', '2021/7/19 1:23:00')
        self.scenario.add_action_points('红方得分100', '红方', 100)
        self.scenario.add_event('事件1')
        self.scenario.set_event_trigger('事件1', '时间触发器')
        self.scenario.set_event_action('事件1', '红方得分100')
        self.env.step()
        score = self.red_side.get_score()
        self.assertEqual(score, 100)

    def test_get_unit_by_guid(self):
        """根据guid获取实体对象"""
        unit = self.red_side.get_unit_by_guid(self.antisubmarine_aircraft_guid)
        self.assertEqual(unit.strGuid, self.antisubmarine_aircraft_guid)

    def test_get_contact_by_guid(self):
        """根据情报对象guid获取情报对象"""
        contact = self.red_side.get_contact_by_guid(self.enemy_airplane_guid)
        self.assertEqual(contact.strGuid, self.enemy_airplane_guid)

    def test_get_identified_targets_by_name(self):
        """从推演方用名称确认目标"""
        contacts = self.red_side.get_identified_targets_by_name('苏-34 型“鸭嘴兽”攻击机')
        for k, v in contacts.items():
            self.assertEqual(v.strName, '苏-34 型“鸭嘴兽”攻击机')

    def test_get_elevation(self):
        """获取某点的海拔高度"""
        height = self.red_side.get_elevation((self.airport_1.dLatitude, self.airport_1.dLongitude))
        self.assertEqual(self.airport_1.fCurrentAltitude_ASL, height)

    def test_add_unit(self):
        """添加单元"""
        self.red_side.add_unit('ship', 'B', 383, 61.4871723614, -17.2522930173)
        self.env.step()
        ships = self.red_side.get_ships()
        flag = False
        for k, v in ships.items():
            if v.strName == 'B':
                self.assertEqual(v.dLatitude, 61.4871723614)
                self.assertEqual(v.dLongitude, -17.2522930173)
                self.assertTrue('383' in v.strDBGUID)
                flag = True
        self.assertTrue(flag)

    def test_add_submarine(self):
        """添加潜艇"""
        self.red_side.add_submarine('S', 32, 61.4871723614997, -17.2522930173048)
        self.env.step()
        subs = self.red_side.get_submarines()
        flag = False
        for k, v in subs.items():
            if v.strName == 'S':
                # 添加的潜艇有默认速度
                # self.assertEqual(v.dLatitude, 61.4871723614997)
                # self.assertEqual(v.dLongitude, -17.2522930173048)
                self.assertTrue('32' in v.strDBGUID)
                flag = True
        self.assertTrue(flag)

    def test_add_ship(self):
        """添加舰船"""
        self.red_side.add_ship('B', 383, 61.487172361499, -17.252293017304)
        self.env.step()
        ships = self.red_side.get_ships()
        flag = False
        for k, v in ships.items():
            if v.strName == 'B':
                flag = True
                self.assertEqual(v.dLatitude, 61.487172361499)
                self.assertEqual(v.dLongitude, -17.252293017304)
                self.assertTrue('383' in v.strDBGUID)
        self.assertTrue(flag)

    def test_add_facility(self):
        """添加地面兵力设施"""
        self.red_side.add_facility('地面兵力', 383, 30.938171538861, 121.06210687030, 30)
        self.env.step()
        facilities = self.red_side.get_facilities()
        flag = False
        for k, v in facilities.items():
            if v.strName == '地面兵力':
                self.assertEqual(v.dLatitude, 30.938171538861)
                self.assertEqual(v.dLongitude, 121.06210687030)
                self.assertTrue('383' in v.strDBGUID)
                flag = True
        self.assertTrue(flag)

    def test_add_aircraft(self):
        """添加飞机"""
        self.red_side.add_aircraft('飞机-添加', 6, 119, 30.9381715388612, 121.062106870304)
        self.env.step()
        airs = self.red_side.get_aircrafts()
        flag = False
        for k, v in airs.items():
            if v.strName == '飞机-添加':
                self.assertTrue('6' in v.strDBGUID)
                self.assertTrue('119' in v.strLoadoutDBGUID)
                # TODO 高度和航向设置不正确
                flag = True
        self.assertTrue(flag)

    def test_add_satellite(self):
        """添加卫星"""
        self.red_side.add_satellite('hsfw-datasatellite-00000000009', 2)
        self.env.step()
        satellites = self.red_side.get_satellites()
        flag = False
        for k, v in satellites.items():
            if v.strName == 'Yaogan-4' or v.strName == '遥感4号卫星':
                flag = True
        self.assertTrue(flag)

    def test_add_mission_patrol_air(self):
        """添加空战巡逻任务"""
        point_list = ['RP-74', 'RP-75', 'RP-76', 'RP-77']
        # 空战巡逻
        obj = self.red_side.add_mission_patrol('空战巡逻', 0, point_list)
        self.env.step()
        for i in range(10):
            self.env.step()
        missions = self.red_side.get_patrol_missions()
        flag = False
        for k, v in missions.items():
            if k == obj.strGuid:
                flag = True
                self.assertEqual(v.m_MissionClass, 2)
                self.assertEqual(v.strName, '空战巡逻')
        self.assertTrue(flag)

    def test_add_mission_patrol_surface(self):
        """添加反水面战巡逻任务"""
        point_list = ['RP-74', 'RP-75', 'RP-76', 'RP-77']
        # 反水面战巡逻
        obj = self.red_side.add_mission_patrol('反水面战巡逻', 1, point_list)
        self.env.step()

        missions = self.red_side.get_patrol_missions()
        flag = False
        for k, v in missions.items():
            if k == obj.strGuid:
                flag = True
                self.assertEqual(v.m_MissionClass, 2)
                self.assertEqual(v.strName, '反水面战巡逻')
        self.assertTrue(flag)

    def test_add_mission_patrol_land(self):
        """添加反地面战巡逻任务"""
        point_list = ['RP-74', 'RP-75', 'RP-76', 'RP-77']
        # 反地面战巡逻
        obj = self.red_side.add_mission_patrol('反地面战巡逻', 2, point_list)
        self.env.step()

        missions = self.red_side.get_patrol_missions()
        flag = False
        for k, v in missions.items():
            if k == obj.strGuid:
                flag = True
                self.assertEqual(v.m_MissionClass, 2)
                self.assertEqual(v.strName, '反地面战巡逻')
        self.assertTrue(flag)

    def test_add_mission_patrol_sur_land(self):
        """添加反水面及地面战巡逻任务"""
        point_list = ['RP-74', 'RP-75', 'RP-76', 'RP-77']
        # 反水面及地面战巡逻
        obj = self.red_side.add_mission_patrol('反水面及地面战巡逻', 3, point_list)
        self.env.step()

        missions = self.red_side.get_patrol_missions()
        flag = False
        for k, v in missions.items():
            if k == obj.strGuid:
                flag = True
                self.assertEqual(v.m_MissionClass, 2)
                self.assertEqual(v.strName, '反水面及地面战巡逻')
        self.assertTrue(flag)

    def test_add_mission_patrol_sub(self):
        """添加反潜巡逻任务"""
        point_list = ['RP-74', 'RP-75', 'RP-76', 'RP-77']
        # 反潜巡逻
        obj = self.red_side.add_mission_patrol('反潜巡逻', 4, point_list)
        self.env.step()

        missions = self.red_side.get_patrol_missions()
        flag = False
        for k, v in missions.items():
            if k == obj.strGuid:
                flag = True
                self.assertEqual(v.m_MissionClass, 2)
                self.assertEqual(v.strName, '反潜巡逻')
        self.assertTrue(flag)

    def test_add_mission_patrol_SEAD(self):
        """添加压制敌防空巡逻任务"""
        point_list = ['RP-74', 'RP-75', 'RP-76', 'RP-77']
        # 压制敌防空巡逻
        obj = self.red_side.add_mission_patrol('压制敌防空巡逻', 5, point_list)
        self.env.step()

        missions = self.red_side.get_patrol_missions()
        flag = False
        for k, v in missions.items():
            if k == obj.strGuid:
                flag = True
                self.assertEqual(v.m_MissionClass, 2)
                self.assertEqual(v.strName, '压制敌防空巡逻')
        self.assertTrue(flag)

    def test_add_mission_patrol_sea(self):
        """添加海上控制巡逻任务"""
        point_list = ['RP-74', 'RP-75', 'RP-76', 'RP-77']
        # 海上控制巡逻
        obj = self.red_side.add_mission_patrol('海上控制巡逻', 6, point_list)
        self.env.step()

        missions = self.red_side.get_patrol_missions()
        flag = False
        for k, v in missions.items():
            if k == obj.strGuid:
                flag = True
                self.assertEqual(v.m_MissionClass, 2)
                self.assertEqual(v.strName, '海上控制巡逻')
        self.assertTrue(flag)

    def test_add_mission_strike_air(self):
        """添加对空拦截任务"""
        # 对空拦截
        obj = self.red_side.add_mission_strike('对空拦截', 0)
        self.env.step()

        missions = self.red_side.get_strike_missions()
        flag = False
        for k, v in missions.items():
            if k == obj.strGuid:
                flag = True
                self.assertEqual(v.m_MissionClass, 1)
                self.assertEqual(v.strName, '对空拦截')
                self.assertEqual(v.m_StrikeType, 0)
        self.assertTrue(flag)

    def test_add_mission_strike_land(self):
        """添加对陆打击任务"""
        # 对陆打击
        obj = self.red_side.add_mission_strike('对陆打击', 1)
        self.env.step()

        missions = self.red_side.get_strike_missions()
        flag = False
        for k, v in missions.items():
            if k == obj.strGuid:
                flag = True
                self.assertEqual(v.m_MissionClass, 1)
                self.assertEqual(v.strName, '对陆打击')
                self.assertEqual(v.m_StrikeType, 1)
        self.assertTrue(flag)

    def test_add_mission_strike_surface(self):
        """添加对海突击任务"""
        # 对海突击
        obj = self.red_side.add_mission_strike('对海突击', 2)
        self.env.step()

        missions = self.red_side.get_strike_missions()
        flag = False
        for k, v in missions.items():
            if k == obj.strGuid:
                flag = True
                self.assertEqual(v.m_MissionClass, 1)
                self.assertEqual(v.strName, '对海突击')
                self.assertEqual(v.m_StrikeType, 2)
        self.assertTrue(flag)

    def test_add_mission_strike_sub(self):
        """添加对潜攻击任务"""
        # 对潜攻击
        obj = self.red_side.add_mission_strike('对潜攻击', 3)
        self.env.step()

        missions = self.red_side.get_strike_missions()
        flag = False
        for k, v in missions.items():
            if k == obj.strGuid:
                flag = True
                self.assertEqual(v.m_MissionClass, 1)
                self.assertEqual(v.strName, '对潜攻击')
                self.assertEqual(v.m_StrikeType, 3)
        self.assertTrue(flag)

    def test_add_mission_support(self):
        """添加支援任务"""
        point_list = ['RP-74', 'RP-75', 'RP-76', 'RP-77']
        obj = self.red_side.add_mission_support('支援任务-新', point_list)
        self.env.step()

        missions = self.red_side.get_support_missions()
        flag = False
        for k, v in missions.items():
            if k == obj.strGuid:
                flag = True
                self.assertEqual(v.m_MissionClass, 3)
                self.assertEqual(v.strName, '支援任务-新')
        self.assertTrue(flag)

    def test_add_mission_ferry(self):
        """添加转场任务"""
        obj = self.red_side.add_mission_ferry('转场任务-新', '机场2')
        obj = self.red_side.add_mission_ferry('转场任务-新2', self.airport_1_guid)
        self.env.step()

        missions = self.red_side.get_ferry_missions()
        flag = False
        for k, v in missions.items():
            if k == obj.strGuid:
                flag = True
                self.assertEqual(v.m_MissionClass, 4)
                self.assertEqual(v.strName, '转场任务-新2')
        self.assertTrue(flag)

    def test_add_mission_mining(self):
        """添加布雷任务"""
        point_list = ['RP-74', 'RP-75', 'RP-76', 'RP-77']
        obj = self.red_side.add_mission_mining('布雷任务-新', point_list)
        self.env.step()
        self.assertEqual(obj.strName, '布雷任务-新')
        self.env.step()
        # # 布雷任务设置飞机航速和高度
        # obj.assign_unit(self.antisubmarine_aircraft_guid)
        # air = self.red_side.get_unit_by_guid(self.antisubmarine_aircraft_guid)
        # air.set_desired_speed(555.6)
        # air.set_desired_height(3000, 'true')
        # # 设置任务时间
        # obj.set_start_time('2021-07-19 22:10:00')
        # # 设置任务状态
        # obj.set_is_active('false')
        # self.env.step()

    def test_add_mission_mine_clearing(self):
        """添加扫雷任务"""
        point_list = ['RP-74', 'RP-75', 'RP-76', 'RP-77']
        obj = self.red_side.add_mission_mine_clearing('扫雷任务-新', point_list)
        self.env.step()
        self.assertEqual(obj.strName, '扫雷任务-新')
        self.env.step()
        # # 扫雷任务设置飞机航速和高度
        # obj.assign_unit(self.antisubmarine_aircraft_2_guid)
        # air = self.red_side.get_unit_by_guid(self.antisubmarine_aircraft_2_guid)
        # air.set_desired_speed(555.6)
        # air.set_desired_height(3000, 'true')
        # # 设置任务时间
        # obj.set_start_time('2021-07-19 22:10:00')
        # # 设置任务状态
        # obj.set_is_active('false')

    def test_add_mission_cargo(self):
        """添加投送任务"""
        point_list = ['RP-74', 'RP-75', 'RP-76', 'RP-77']
        obj = self.red_side.add_mission_cargo('投送任务-新', point_list)
        self.env.step()

        missions = self.red_side.get_cargo_missions()
        flag = False
        for k, v in missions.items():
            if k == obj.strGuid:
                flag = True
                self.assertEqual(v.m_MissionClass, 8)
                self.assertEqual(v.strName, '投送任务-新')
        self.assertTrue(flag)

    def test_delete_mission(self):
        """删除任务"""
        point_list = ['RP-74', 'RP-75', 'RP-76']
        obj = self.red_side.add_mission_cargo('投送任务-新', point_list)
        self.env.step()
        self.red_side.delete_mission('投送任务-新')
        self.env.step()

        missions = self.red_side.get_cargo_missions()
        flag = False
        for k, v in missions.items():
            if k == obj.strGuid:
                flag = True
        self.assertFalse(flag)

    def test_add_group(self):
        """将同类型单元单元合并创建编队，"""
        unit_list = [self.antisubmarine_aircraft_guid, self.antisubmarine_aircraft_2_guid]
        return_obj = self.red_side.add_group(unit_list)
        self.env.step()
        groups = self.red_side.get_groups()
        flag = False
        for k, v in groups.items():
            if k == return_obj.strGuid:
                flag = True
                self.assertEqual(return_obj.strName, v.strName)
        self.assertTrue(flag)

    def test_air_group_out(self):
        """飞机编组出动"""
        groups = self.red_side.get_groups()
        group_len = len(groups)
        unit_list = [self.docked_f35_1_guid, self.docked_f35_2_guid]
        self.red_side.air_group_out(unit_list)
        self.env.step()
        groups_new = self.red_side.get_groups()
        group_len_new = len(groups_new)
        self.assertEqual(group_len + 1, group_len_new)

    def test_add_unit_to_facility(self):
        """往机场，码头添加单元"""
        return_obj_1 = self.red_side.add_unit_to_facility(unit_type='air', name='基地飞机-新', dbid=13,
                                                          base_unit_guid=self.airport_1_guid, loadout_id=8063)
        return_obj_2 = self.red_side.add_unit_to_facility(unit_type='ship', name='基地舰船-新', dbid=4,
                                                          base_unit_guid=self.wharf_1_guid)
        return_obj_3 = self.red_side.add_unit_to_facility(unit_type='sub', name='基地潜艇-新', dbid=5,
                                                          base_unit_guid=self.wharf_1_guid)
        self.env.step()
        aircraft = self.red_side.get_unit_by_guid(return_obj_1.strGuid)
        self.assertEqual(aircraft.strName, '基地飞机-新')
        self.assertEqual(aircraft.m_HostActiveUnit, self.airport_1_guid)

    def test_delete_all_unit(self):
        """删除本方所有单元"""
        self.red_side.delete_all_unit()
        self.env.step()
        aircrafts = self.red_side.get_aircrafts()
        self.assertFalse(aircrafts)

    def test_set_ecom_status(self):
        """设置选定对象的 EMCON"""
        self.red_side.set_ecom_status('Side', '红方', 'Radar=Active')
        self.red_side.set_ecom_status('Side', '红方', 'Sonar=Active')
        self.red_side.set_ecom_status('Side', '红方', 'OECM=Active')
        self.env.step()
        doctrine = self.red_side.get_doctrine()
        self.assertEqual(doctrine.m_EMCON_SettingsForOECM, 1)
        self.assertEqual(doctrine.m_EMCON_SettingsForRadar, 1)
        self.assertEqual(doctrine.m_EMCON_SettingsForSonar, 1)

        mission = self.red_side.get_missions_by_name('空中巡逻')
        self.red_side.set_ecom_status('Mission', mission.strGuid, 'Radar=Passive')
        self.red_side.set_ecom_status('Mission', mission.strGuid, 'Sonar=Passive')
        self.red_side.set_ecom_status('Mission', mission.strGuid, 'OECM=Passive')
        self.env.step()
        doctrine_mission = mission.get_doctrine()
        self.assertEqual(doctrine_mission.m_EMCON_SettingsForOECM, 0)
        self.assertEqual(doctrine_mission.m_EMCON_SettingsForRadar, 0)
        self.assertEqual(doctrine_mission.m_EMCON_SettingsForSonar, 0)

        groups = self.red_side.get_groups()
        group = common.get_obj_by_name(groups, '飞行编队 37')
        self.red_side.set_ecom_status('Group', group.strGuid, 'Radar=Passive')
        self.red_side.set_ecom_status('Group', group.strGuid, 'Sonar=Passive')
        self.red_side.set_ecom_status('Group', group.strGuid, 'OECM=Passive')
        self.env.step()
        doctrine_group = group.get_doctrine()
        self.assertEqual(doctrine_group.m_EMCON_SettingsForOECM, 0)
        self.assertEqual(doctrine_group.m_EMCON_SettingsForRadar, 0)
        self.assertEqual(doctrine_group.m_EMCON_SettingsForSonar, 0)

        self.red_side.set_ecom_status('Unit', self.ship.strGuid, 'Radar=Passive')
        self.red_side.set_ecom_status('Unit', self.ship.strGuid, 'Sonar=Passive')
        self.red_side.set_ecom_status('Unit', self.ship.strGuid, 'OECM=Passive')
        self.env.step()
        doctrine_unit = self.ship.get_doctrine()
        self.assertEqual(doctrine_unit.m_EMCON_SettingsForOECM, 0)
        self.assertEqual(doctrine_unit.m_EMCON_SettingsForRadar, 0)
        self.assertEqual(doctrine_unit.m_EMCON_SettingsForSonar, 0)

        self.red_side.set_ecom_status('Unit', self.ship.strGuid, 'Inherit')
        self.env.step()
        doctrine_unit = self.ship.get_doctrine()
        self.assertEqual(doctrine_unit.m_bEMCON_AccordingSuperior, True)

    def test_add_reference_point(self):
        """添加参考点"""
        point = self.red_side.add_reference_point('点1', 27.6642320262197, 134.621472742701)
        self.env.step()
        self.assertEqual(point.strName, '点1')
        self.assertEqual(point.dLatitude, 27.6642320262197)
        self.assertEqual(point.dLongitude, 134.621472742701)

    def test_add_zone(self):
        """创建区域"""
        area = ['RP-66', 'RP-67', 'RP-68', 'RP-69']
        affects = ['Aircraft', 'Ship', 'Submarine']
        zone_1 = self.red_side.add_zone(zone_type=0, description='禁航区-新', area=area, affects=affects, is_active='true')
        zone_2 = self.red_side.add_zone(zone_type=1, description='封锁区-新', area=area, affects=affects, is_active='true',
                                        mark_as='Hostile')
        zone_3 = self.red_side.add_zone(zone_type=1, description='封锁区-新2', area=area, affects=affects,
                                        is_active='false',
                                        mark_as='Unfriendly')
        self.env.step()
        exclusion_zones = self.red_side.get_exclusion_zones()
        no_nav_zones = self.red_side.get_no_nav_zones()

        zone_1 = common.get_obj_by_desc(no_nav_zones, '禁航区-新')
        self.assertEqual(zone_1.bIsActive, True)
        self.assertEqual(zone_1.m_AffectedUnitTypes, '1@2@3')
        for item in area:
            rp = self.red_side.get_reference_point_by_name(item)
            self.assertTrue(rp.strName in area)

        zone_2 = common.get_obj_by_desc(exclusion_zones, '封锁区-新')
        self.assertEqual(zone_2.bIsActive, True)
        self.assertEqual(zone_2.m_MarkViolatorAs, 3)
        self.assertEqual(zone_2.m_AffectedUnitTypes, '1@2@3')
        for item in area:
            rp = self.red_side.get_reference_point_by_name(item)
            self.assertTrue(rp.strName in area)

        zone_3 = common.get_obj_by_desc(exclusion_zones, '封锁区-新2')
        self.assertEqual(zone_3.bIsActive, False)
        self.assertEqual(zone_3.m_MarkViolatorAs, 2)
        self.assertEqual(zone_3.m_AffectedUnitTypes, '1@2@3')
        for item in area:
            rp = self.red_side.get_reference_point_by_name(item)
            self.assertTrue(rp.strName in area)

    def test_set_zone(self):
        """设置区域"""
        area = ['RP-66', 'RP-67', 'RP-68', 'RP-69']
        affects = ['Aircraft', 'Ship', 'Submarine']
        zone_2 = self.red_side.add_zone(zone_type=1, description='封锁区-新', area=area, affects=affects, is_active='true',
                                        mark_as='Hostile')
        area = ['RP-66', 'RP-67', 'RP-68']
        affects = ['Aircraft', 'Ship']
        self.red_side.set_zone(zone_guid=zone_2.strGuid, description='封锁区-改', area=area, affects=affects,
                               is_active='false', mark_as='Unfriendly', rp_visible='false')
        self.red_side.set_zone(zone_guid=zone_2.strGuid, affects=affects)
        self.env.step()

        exclusion_zones = self.red_side.get_exclusion_zones()
        zone_3 = common.get_obj_by_desc(exclusion_zones, '封锁区-改')
        self.assertEqual(zone_3.bIsActive, False)
        self.assertEqual(zone_3.m_MarkViolatorAs, 2)
        self.assertEqual(zone_3.m_AffectedUnitTypes, '1@2')
        for item in area:
            rp = self.red_side.get_reference_point_by_name(item)
            self.assertTrue(rp.strName in area)

    def test_deploy_mine(self):
        """给某一方添加雷"""
        area = ['RP-66', 'RP-67', 'RP-68', 'RP-69']
        self.red_side.deploy_mine('hsfw-dataweapon-00000000000634', 2, area)
        self.env.step()
        weapons_1 = self.red_side.get_unguided_weapons()
        count = 0
        for k, v in weapons_1.items():
            count += 1
            self.assertEqual(v.strDBGUID, 'hsfw-dataweapon-00000000000634')
        self.assertEqual(count, 2)

    def test_set_new_name(self):
        """推演方重命名"""
        self.red_side.set_new_name('绿方')
        self.env.step()
        self.assertEqual(self.red_side.strName, '绿方')

    def test_set_mark_contact(self):
        """设置目标对抗关系"""
        self.red_side.set_mark_contact(self.enemy_airplane_guid, 'H')
        self.env.step()
        contacts = self.red_side.get_contacts()
        contact = contacts.get(self.enemy_airplane_guid)
        posture_dict = self.get_side_posture_dict(contact.m_SidePostureStanceDictionary)
        self.assertEqual(posture_dict[self.red_side.strGuid], 3)

    @staticmethod
    def get_side_posture_dict(side_posture_stance_dictionary):
        dict1 = {}
        if side_posture_stance_dictionary:
            str_list = side_posture_stance_dictionary.split('@')
            for item in str_list:
                if item:
                    item_list = item.split('$')
                    dict1[item_list[0]] = int(item_list[1])
        return dict1

    def test_assign_target_to_mission(self):
        """将目标分配给一项打击任务"""
        strike_mission = self.red_side.add_mission_strike('对空拦截-新', 0)
        self.red_side.assign_target_to_mission(self.enemy_airplane_guid, strike_mission.strGuid)
        self.red_side.assign_target_to_mission(self.enemy_airplane_guid_2, strike_mission.strName)
        self.env.step()
        strike_mission = self.red_side.get_missions_by_name('对空拦截-新')
        self.assertTrue(self.enemy_airplane_guid in strike_mission.m_SpecificTargets)
        self.assertTrue(self.enemy_airplane_guid_2 in strike_mission.m_SpecificTargets)

    def test_set_score(self):
        """设置推演方总分"""
        self.red_side.set_score(100, '测试用')
        self.env.step()
        self.assertEqual(self.red_side.get_score(), 100)

    def test_side_scoring(self):
        """设置完胜完败阈值"""
        # TODO 设置后服务端蓝方评分规则变化，没有对应的态势更新
        self.red_side.side_scoring(-100, 100)
        self.env.step()

    def test_drop_contact(self):
        """放弃目标, 不再将所选目标列为探测对象"""

        # 设置对空限制开火
        side_doctrine = self.red_side.get_doctrine()
        side_doctrine.set_weapon_control_status('weapon_control_status_air', 2)

        # 向敌机1发射一枚 MIM-104B型“爱国者-1”防空导弹
        weapon_guid = 'hsfw-dataweapon-00000000001152'
        self.ground_to_air_missile_squadron.allocate_weapon_to_target(self.enemy_airplane_guid, weapon_guid, 1)

        self.red_side.drop_contact(self.enemy_airplane_guid)
        self.env.step()

        # 获取所有武器
        weapons = self.red_side.get_weapons()
        flag = False
        for k, v in weapons.items():
            if 'MIM-104B型“爱国者-1”防空导弹' in v.strName:
                flag = True
                break
        self.assertEqual(False, flag)

    def test_wcsfa_contact_types_all_unit(self):
        """控制所有单元对所有目标类型的攻击状态"""
        doctrine = self.ship.get_doctrine()
        self.red_side.wcsfa_contact_types_all_unit('Hold')
        self.env.step()
        self.assertEqual(2, doctrine.m_WCS_Surface)
        self.assertEqual(2, doctrine.m_WCS_Land)
        self.assertEqual(2, doctrine.m_WCS_Air)
        self.assertEqual(2, doctrine.m_WCS_Submarine)

        self.red_side.wcsfa_contact_types_all_unit('Tight')
        self.env.step()
        self.assertEqual(1, doctrine.m_WCS_Surface)
        self.assertEqual(1, doctrine.m_WCS_Land)
        self.assertEqual(1, doctrine.m_WCS_Air)
        self.assertEqual(1, doctrine.m_WCS_Submarine)

        self.red_side.wcsfa_contact_types_all_unit('Free')
        self.env.step()
        self.assertEqual(0, doctrine.m_WCS_Surface)
        self.assertEqual(0, doctrine.m_WCS_Land)
        self.assertEqual(0, doctrine.m_WCS_Air)
        self.assertEqual(0, doctrine.m_WCS_Submarine)

        self.red_side.wcsfa_contact_types_all_unit('Inherited')
        self.env.step()
        self.assertEqual(-1, doctrine.m_WCS_Surface)
        self.assertEqual(-1, doctrine.m_WCS_Land)
        self.assertEqual(-1, doctrine.m_WCS_Air)
        self.assertEqual(-1, doctrine.m_WCS_Submarine)

    def test_lpcw_attack_all_unit(self):
        """所有单元攻击时是否忽略计划航线"""
        doctrine = self.ship.get_doctrine()

        self.red_side.lpcw_attack_all_unit('Yes')
        self.env.step()
        self.assertEqual(1, doctrine.m_IgnorePlottedCourseWhenAttacking)

        self.red_side.lpcw_attack_all_unit('No')
        self.env.step()
        self.assertEqual(0, doctrine.m_IgnorePlottedCourseWhenAttacking)

        self.red_side.lpcw_attack_all_unit('Inherited')
        self.env.step()
        self.assertEqual(-1, doctrine.m_IgnorePlottedCourseWhenAttacking)

    def test_set_side_options(self):
        """设置认知能力、训练水平、AI 操控、集体反应、自动跟踪非作战单元等组成的属性集合"""
        awareness = 'Blind'
        proficiency = 'Novice'
        is_ai_only = 'true'
        is_coll_response = 'true'
        is_auto_track_civs = 'true'
        self.red_side.set_side_options(awareness=awareness, proficiency=proficiency, is_ai_only=is_ai_only,
                                       is_coll_response=is_coll_response, is_auto_track_civs=is_auto_track_civs)
        self.env.step()
        self.assertEqual(self.red_side.m_ProficiencyLevel, 0)
        self.assertEqual(self.red_side.m_AwarenessLevel, -1)
        self.assertEqual(self.red_side.bAIOnly, True)
        self.assertEqual(self.red_side.bCATC, True)
        self.assertEqual(self.red_side.bCollectiveResponsibility, True)
        awareness = 'OMNI'
        proficiency = 'Ace'
        is_ai_only = 'false'
        is_coll_response = 'false'
        is_auto_track_civs = 'false'
        self.red_side.set_side_options(awareness=awareness, proficiency=proficiency, is_ai_only=is_ai_only,
                                       is_coll_response=is_coll_response, is_auto_track_civs=is_auto_track_civs)
        self.env.step()
        self.assertEqual(self.red_side.m_ProficiencyLevel, 4)
        self.assertEqual(self.red_side.m_AwarenessLevel, 3)
        self.assertEqual(self.red_side.bAIOnly, False)
        self.assertEqual(self.red_side.bCATC, False)
        self.assertEqual(self.red_side.bCollectiveResponsibility, False)

    def test_get_side_options(self):
        """获取推演方属性"""
        side_options = self.red_side.get_side_options()
        self.env.step()
        self.assertEqual(side_options['guid'], self.red_side.strGuid)
        self.assertEqual(side_options['side'], self.red_side.strName)
        self.assertEqual(side_options['proficiency'], 'Regular')
        self.assertEqual(side_options['awareness'], 'Normal')

    def test_get_side_is_human(self):
        """获取推演方操控属性"""
        is_human = self.red_side.get_side_is_human()
        self.env.step()
        self.assertEqual(is_human, 'Yes')

    def test_copy_unit(self):
        """将想定中当前推演方中的已有单元复制到指定经纬度处"""
        obj = self.red_side.copy_unit('反潜机1', 28.0, 28.0)
        self.env.step()
        aircrafts = self.red_side.get_aircrafts()
        count = 0
        for k, v in aircrafts.items():
            if v.strName == '反潜机1':
                count += 1
        self.assertEqual(count, 2)

    def test_delete_unit(self):
        """删除当前推演方中指定单元"""
        obj = self.red_side.delete_unit('反潜机1')
        self.env.step()
        aircrafts = self.red_side.get_aircrafts()
        flag = False
        for k, v in aircrafts.items():
            if v.strName == '反潜机1':
                flag = True
        self.assertFalse(flag)

    def test_delete_group(self):
        """删除编组"""
        self.red_side.delete_group('飞行编队 37', 'true')
        self.env.step()
        groups = self.red_side.get_groups()
        flag = False
        for k, v in groups.items():
            if v.strName == '飞行编队 37':
                flag = True
        self.assertFalse(flag)

        aircrafts = self.red_side.get_aircrafts()
        flag = False
        for k, v in aircrafts.items():
            if v.strName == '反潜机3':
                flag = True
        self.assertFalse(flag)

    def test_delete_group_2(self):
        """解散编组"""
        self.red_side.delete_group('飞行编队 37')
        self.env.step()
        groups = self.red_side.get_groups()
        flag = False
        for k, v in groups.items():
            if v.strName == '飞行编队 37':
                flag = True
        self.assertFalse(flag)

        aircrafts = self.red_side.get_aircrafts()
        flag = False
        for k, v in aircrafts.items():
            if v.strName == '反潜机3':
                flag = True
        self.assertTrue(flag)

    def test_kill_unit(self):
        """摧毁单元"""
        self.red_side.kill_unit('反潜机1')
        self.env.step()
        aircrafts = self.red_side.get_aircrafts()
        flag = False
        for k, v in aircrafts.items():
            if v.strName == '反潜机1':
                flag = True
        self.assertFalse(flag)
        # 红方战损记录增加

    def test_remove_zone(self):
        """删除指定推演方的指定禁航区或封锁区"""
        area = ['RP-66', 'RP-67', 'RP-68', 'RP-69']
        affects = ['Aircraft', 'Ship', 'Submarine']
        zone = self.red_side.add_zone(zone_type=1, description='封锁区-新', area=area, affects=affects, is_active='true',
                                      mark_as='Hostile')
        self.env.step()
        self.red_side.remove_zone(zone.strGuid)
        self.env.step()
        zones = self.red_side.get_exclusion_zones()
        flag = False
        for k, v in zones.items():
            if v.strName == '封锁区-新':
                flag = True
        self.assertFalse(flag)

    def test_delete_reference_point(self):
        """删除参考点"""
        point = self.red_side.add_reference_point('点1', 27.6642320262197, 134.621472742701)
        self.env.step()
        self.red_side.delete_reference_point(point.strGuid)
        self.env.step()
        points = self.red_side.get_reference_points()
        flag = False
        for k, v in points.items():
            if v.strName == '点1':
                flag = True
        self.assertFalse(flag)

    def test_delete_reference_point_by_name(self):
        """按参考点名称删除参考点"""
        point = self.red_side.add_reference_point('RP-XX', 27.6642320262197, 134.621472742701)
        self.env.step()
        self.red_side.delete_reference_point_by_name('RP-XX')
        self.env.step()
        points = self.red_side.get_reference_points()
        flag = False
        for k, v in points.items():
            if v.strName == 'RP-XX':
                flag = True
        self.assertFalse(flag)

    def test_add_plan_way(self):
        """为指定推演方添加一预设航线"""
        self.red_side.add_plan_way(0, '单元航线-新')
        self.red_side.add_plan_way(1, '武器航线-新')
        self.env.step()
        sideways = self.red_side.get_sideways()
        side_way_unit = common.get_obj_by_name(sideways, '单元航线-新')
        self.assertEqual(side_way_unit.m_eSideWayType, 0)
        side_way_weapon = common.get_obj_by_name(sideways, '武器航线-新')
        self.assertEqual(side_way_weapon.m_eSideWayType, 1)

    def test_set_plan_way_showing_status(self):
        """控制预设航线的显示或隐藏"""
        self.red_side.add_plan_way(0, '单元航线-新')
        self.env.step()
        sideways = self.red_side.get_sideways()
        side_way_unit = common.get_obj_by_name(sideways, '单元航线-新')
        self.red_side.set_plan_way_showing_status('单元航线-新', 'false')
        self.env.step()
        self.assertEqual(side_way_unit.m_bShow, False)
        self.red_side.set_plan_way_showing_status('单元航线-新', 'true')
        self.env.step()
        self.assertEqual(side_way_unit.m_bShow, True)

    def test_rename_plan_way(self):
        """修改预设航线的名称"""
        self.red_side.add_plan_way(0, '单元航线-新')
        self.red_side.rename_plan_way('单元航线-新', '单元航线-改')
        self.env.step()
        sideways = self.red_side.get_sideways()
        side_way_unit = common.get_obj_by_name(sideways, '单元航线-改')
        self.assertTrue(side_way_unit)

    def test_add_plan_way_point(self):
        """为预设航线添加航路点"""
        self.env.step()
        self.red_side.add_plan_way(0, '单元航线-新')
        self.red_side.add_plan_way_point('单元航线-新', 28.0, 29.0)
        self.red_side.add_plan_way_point('单元航线-新', 29.0, 30.0)
        self.env.step()
        sideways = self.red_side.get_sideways()
        side_way_unit = common.get_obj_by_name(sideways, '单元航线-新')

        way_point_guids = side_way_unit.m_WayPoints.split('@')
        way_points = self.red_side.situation.waypoint_dic
        self.assertEqual(len(way_point_guids), 2)
        for item in way_point_guids:
            way_point = way_points.get(item)
            self.assertTrue(way_point.dLatitude in [29.0, 30.0])
            self.assertTrue(way_point.dLongitude in [28.0, 29.0])

    def test_remove_plan_way(self):
        """删除预设航线"""
        self.red_side.add_plan_way(0, '单元航线-新')
        self.red_side.remove_plan_way('单元航线-新')
        self.env.step()
        sideways = self.red_side.get_sideways()
        side_way_unit = common.get_obj_by_name(sideways, '单元航线-新')
        self.assertFalse(side_way_unit)

    def test_edit_brief(self):
        """修改指定推演方的任务简报"""
        self.env.step()
        self.red_side.edit_brief('简报-红方')
        self.env.step()
        # TODO 无相关态势， strBriefing为空

    def test_is_target_existed(self):
        """检查目标是否存在"""
        is_exist = self.red_side.is_target_existed('F-14E型“超级雄猫”战斗机')
        self.assertTrue(is_exist)

    def test_hold_position_all_units(self):
        """保持所有单元阵位"""
        self.red_side.hold_position_all_units('true')
        self.env.step()
        # 选中单元，右键，保持阵位--所有单元已勾选
        self.assertTrue(self.ground_to_air_missile_squadron.bHoldPosition)
        self.red_side.hold_position_all_units('false')
        self.env.step()
        # 选中单元，右键，保持阵位--所有单元取消勾选
        self.assertFalse(self.ground_to_air_missile_squadron.bHoldPosition)

    def test_launch_units_in_group(self):
        """编组出航"""
        group_count = len(self.red_side.get_groups())
        self.red_side.launch_units_in_group([self.docked_ship_1, self.docked_ship_2])
        self.env.step()
        group_count_2 = len(self.red_side.get_groups())
        self.assertEqual(group_count_2, group_count + 1)

    def test_launch_units_abort(self):
        """终止出航"""
        group_count = len(self.red_side.get_groups())
        self.red_side.launch_units_in_group([self.docked_ship_1, self.docked_ship_2])
        self.red_side.launch_units_abort([self.docked_ship_1, self.docked_ship_2])
        self.env.step()
        group_count_2 = len(self.red_side.get_groups())
        self.assertEqual(group_count_2, group_count)


if __name__ == '__main__':
    TestSide.main()
