# -*- coding:utf-8 -*-
##########################################################################################################
# File name : submarine.py
# Create date : 2020-1-8
# Modified date : 2020-1-8
# All rights reserved:北京华戍防务技术有限公司
# Author:xy
##########################################################################################################

from mozi_simu_sdk.activeunit import CActiveUnit


class CSubmarine(CActiveUnit):
    """
    潜艇
    """

    def __init__(self, strGuid, mozi_server, situation):
        super().__init__(strGuid, mozi_server, situation)
        self.m_BearingType = {}  # 方位类型
        self.m_Bearing = {}  # 方位
        self.m_Distance = 0.0  # 距离（转换为千米）
        self.bSprintAndDrift = False  # 高低速交替航行
        self.m_AITargets = {}  # 获取AI对象的目标集合
        self.m_AITargetsCanFiretheTargetByWCSAndWeaponQty = {}  # 获取活动单元AI对象的每个目标对应显示不同的颜色集合
        self.strDockAircraft = ''  # 载机按钮的文本描述
        self.strDockShip = ''  # 载艇按钮的文本描述
        # 以下为 CSubmarine 的属性
        self.m_Category = {}  # 类型类别
        self.m_CIC = {}  # 指挥部
        self.m_Rudder = {}  # 船舵
        self.m_PressureHull = {}  # 船身
        # 获取作战单元燃油信息
        self.strFuelState = ''  # 显示燃油状态
        # 柴油剩余百分比
        self.dPercentageDiesel = 0.0
        # 电池剩余百分比
        self.dPercentageBattery = 0.0
        # AIP剩余百分比
        self.dPercentageAIP = 0.0
        self.m_Type = {}
        self.strCavitation = ''
        self.fHoverSpeed = 0.0  # 悬停
        self.fLowSpeed = 0.0  # 低速
        self.fCruiseSpeed = 0.0  # 巡航
        self.fMilitarySpeed = 0.0  # 军力
        self.fAddForceSpeed = 0.0  # 加速
        self.iThermoclineUpDepth = 0.0  # 温跃层上
        self.iThermoclineDownDepth = 0.0  # 温跃层下
        # 载艇-信息
        self.strDamageInfo = ''  # 毁伤
        self.strWeaponInfo = ''  # 武器
        self.strMagazinesInfo = ''  # 弹药库
        self.strFuelInfo = ''  # 燃料
        self.strStatusInfo = ''  # 状态
        self.strTimeToReadyInfo = ''  # 就绪时间
        # 油门高度-航路点信息
        self.strWayPointName = ''  # 航路点名称
        self.ClassName = 'CSubmarine'

    def set_desired_height(self, desired_height, moveto='true'):
        """
        功能：设置潜艇的期望高度
        限制：专项赛限制使用，禁止设置moveto='false'
        参数：desired_height {int or float, 期望潜艇深度值, 为正}
            moveto {str, 'true'-是，瞬间到达该高度, 'false'-否，不瞬间到达该高度}
        返回：'lua执行成功' 或 '脚本执行出错'
        作者：张志高
        单位：北京华戍防务技术有限公司
        时间：2022-1-6
        """
        if isinstance(desired_height, int) or isinstance(desired_height, float):
            lua_script = "ScenEdit_SetUnit({guid='" + str(self.strGuid) + "', depth='" + str(
                desired_height) + "', moveto='" + moveto + "'}) "
            return self.mozi_server.send_and_recv(lua_script)
        else:
            pass
