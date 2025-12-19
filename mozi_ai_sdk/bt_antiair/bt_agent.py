# -*- coding:utf-8 -*-
# File name : bt_agent.py
# Create date : 2020/7/20
# All rights reserved:北京华戍防务技术有限公司
# Author: Dixit
# Modified by:卡巴司机

from mozi_ai_sdk.bt_antiair.leaf_nodes import *
from mozi_ai_sdk.btmodel.bt.bt_nodes import BT


class CAgent:
    def __init__(self):
        self.class_name = 'bt_'
        self.bt = None
        self.nonavbt = None

    def init_bt(self, env, side_name, lenAI, options):
        side = env.scenario.get_side_by_name(side_name)
        sideGuid = side.strGuid
        shortSideKey = "a" + str(lenAI + 1)
        attributes = options
        # 行为树的节点
        hxSequence = BT()
        SAMActivateCondition = BT()
        SAMActivateStatus = BT()
        SAMNotActivated = BT()
        ActivateSAM = BT()

        # 连接节点形成树
        hxSequence.add_child(SAMActivateCondition)
        hxSequence.add_child(SAMActivateStatus)
        SAMActivateStatus.add_child(SAMNotActivated)
        SAMActivateStatus.add_child(ActivateSAM)

        # 每个节点执行的动作
        hxSequence.set_action(hxSequence.sequence, sideGuid, shortSideKey, attributes)
        SAMActivateCondition.set_action(sam_activate_condition, sideGuid, shortSideKey, attributes)
        SAMActivateStatus.set_action(SAMActivateStatus.sequence, sideGuid, shortSideKey, attributes)
        SAMNotActivated.set_action(is_sam_not_activated, sideGuid, shortSideKey, attributes)
        ActivateSAM.set_action(activate_sam, sideGuid, shortSideKey, attributes)
        self.bt = hxSequence

    # 更新行为树
    def update_bt(self, side_name, scenario):
        return self.bt.run(side_name, scenario)
