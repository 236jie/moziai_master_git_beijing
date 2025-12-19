# -*- coding:utf-8 -*-
# File name : leaf_nodes.py
# Create date : 2022/8/10
# All rights reserved:北京华戍防务技术有限公司
# Author:卡巴司机
# Version: 1.0.2
from mozi_utils import geo

sam_guid = '763a105c-f4aa-4be7-b244-5e71d5e98be1'


# 地空导弹激活条件,敌机接近到60km内激活条件变为True
def sam_activate_condition(side_name, scenario):
    side = scenario.get_side_by_name(side_name)
    sam = side.get_unit_by_guid(sam_guid)
    flag = False
    for guid, obj in side.contacts.items():
        dist = geo.get_two_point_distance(
            sam.dLongitude, sam.dLatitude, obj.dLongitude, obj.dLatitude)
        if dist <= 60000.0:
            flag = True
    return flag


# 判断地空导弹是否未激活
def is_sam_not_activated(side_name, scenario):
    side = scenario.get_side_by_name(side_name)
    sam = side.get_unit_by_guid(sam_guid)
    sam_doct = sam.get_doctrine()
    flag = True
    if sam_doct.m_WCS_Air == 0:
        flag = False
    return flag


# 激活地空导弹
def activate_sam(side_name, scenario):
    side = scenario.get_side_by_name(side_name)
    sam = side.get_unit_by_guid(sam_guid)
    sam_doct = sam.get_doctrine()
    sam_doct.set_weapon_control_status('weapon_control_status_air', '0')
    sam_doct.set_em_control_status('Radar', 'Active')
    return True
