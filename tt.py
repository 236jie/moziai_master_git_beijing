import json
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """消息类型枚举"""
    FULL_FRAME = "full_frame"  # 全量帧
    UPDATE_FRAME = "update_frame"  # 增量更新帧
    EVENT = "event"  # 事件
    LOG = "log"  # 日志
    WEAPON_FIRE = "weapon_fire"  # 武器开火
    WEAPON_DESTROYED = "weapon_destroyed"  # 武器被摧毁
    UNIT_STATUS = "unit_status"  # 单位状态
    TERRAIN_CHANGE = "terrain_change"  # 地形变化
    COMMUNICATION = "communication"  # 通信消息


class Priority(Enum):
    """消息优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SimulationMessage:
    """仿真消息数据结构"""
    msg_type: MessageType
    timestamp: float
    game_id: str
    sequence_id: int
    data: Dict[str, Any]
    priority: Priority = Priority.NORMAL
    message_id: str = ""

    def __post_init__(self):
        if not self.message_id:
            self.message_id = f"{self.game_id}_{self.sequence_id}_{self.timestamp}"


@dataclass
class SimulationState:
    """完整的仿真状态"""
    game_id: str = ""
    timestamp: float = 0.0
    sequence_id: int = 0

    # 核心数据结构
    units: Dict[str, Dict] = field(default_factory=dict)  # 单位信息
    weapons: Dict[str, Dict] = field(default_factory=dict)  # 武器信息
    events: List[Dict] = field(default_factory=list)  # 事件列表
    logs: List[Dict] = field(default_factory=list)  # 日志列表
    terrain: Dict[str, Any] = field(default_factory=dict)  # 地形信息
    communications: List[Dict] = field(default_factory=list)  # 通信记录

    # 统计信息
    stats: Dict[str, Any] = field(default_factory=dict)


class DataParser(ABC):
    """数据解析器基类"""

    @abstractmethod
    def can_handle(self, msg_type: MessageType) -> bool:
        """判断是否能处理该类型消息"""
        pass

    @abstractmethod
    def parse_full_frame(self, data: Dict[str, Any], state: SimulationState) -> bool:
        """解析全量帧数据"""
        pass

    @abstractmethod
    def parse_update_frame(self, data: Dict[str, Any], state: SimulationState) -> bool:
        """解析增量更新数据"""
        pass

    @abstractmethod
    def get_priority(self) -> Priority:
        """获取解析器优先级"""
        pass


class UnitStatusParser(DataParser):
    """单位状态解析器"""

    def can_handle(self, msg_type: MessageType) -> bool:
        return msg_type in [MessageType.FULL_FRAME, MessageType.UPDATE_FRAME, MessageType.UNIT_STATUS]

    def parse_full_frame(self, data: Dict[str, Any], state: SimulationState) -> bool:
        """解析全量单位数据"""
        try:
            if 'units' in data:
                state.units.clear()
                for unit_data in data['units']:
                    unit_id = unit_data.get('id')
                    if unit_id:
                        state.units[unit_id] = {
                            'id': unit_id,
                            'type': unit_data.get('type', 'unknown'),
                            'position': unit_data.get('position', {'x': 0, 'y': 0, 'z': 0}),
                            'status': unit_data.get('status', 'active'),
                            'health': unit_data.get('health', 100),
                            'fuel': unit_data.get('fuel', 100),
                            'ammunition': unit_data.get('ammunition', {}),
                            'last_update': time.time()
                        }
                logger.info(f"解析全量单位数据: {len(state.units)} 个单位")
                return True
        except Exception as e:
            logger.error(f"解析单位全量数据失败: {e}")
        return False

    def parse_update_frame(self, data: Dict[str, Any], state: SimulationState) -> bool:
        """解析单位增量更新"""
        try:
            updated_count = 0
            if 'unit_updates' in data:
                for update in data['unit_updates']:
                    unit_id = update.get('id')
                    if unit_id and unit_id in state.units:
                        # 更新现有单位
                        unit = state.units[unit_id]
                        for key, value in update.items():
                            if key != 'id':
                                if key == 'position' and isinstance(value, dict):
                                    unit['position'].update(value)
                                else:
                                    unit[key] = value
                        unit['last_update'] = time.time()
                        updated_count += 1
                    elif unit_id:
                        # 新增单位
                        state.units[unit_id] = {
                            'id': unit_id,
                            'type': update.get('type', 'unknown'),
                            'position': update.get('position', {'x': 0, 'y': 0, 'z': 0}),
                            'status': update.get('status', 'active'),
                            'health': update.get('health', 100),
                            'fuel': update.get('fuel', 100),
                            'ammunition': update.get('ammunition', {}),
                            'last_update': time.time()
                        }
                        updated_count += 1

            # 处理单位删除
            if 'unit_removals' in data:
                for unit_id in data['unit_removals']:
                    if unit_id in state.units:
                        del state.units[unit_id]
                        updated_count += 1

            if updated_count > 0:
                logger.debug(f"更新了 {updated_count} 个单位")
            return updated_count > 0
        except Exception as e:
            logger.error(f"解析单位更新数据失败: {e}")
        return False

    def get_priority(self) -> Priority:
        return Priority.HIGH


class WeaponParser(DataParser):
    """武器系统解析器"""

    def can_handle(self, msg_type: MessageType) -> bool:
        return msg_type in [MessageType.WEAPON_FIRE, MessageType.WEAPON_DESTROYED,
                            MessageType.FULL_FRAME, MessageType.UPDATE_FRAME]

    def parse_full_frame(self, data: Dict[str, Any], state: SimulationState) -> bool:
        try:
            if 'weapons' in data:
                state.weapons.clear()
                for weapon_data in data['weapons']:
                    weapon_id = weapon_data.get('id')
                    if weapon_id:
                        state.weapons[weapon_id] = {
                            'id': weapon_id,
                            'type': weapon_data.get('type'),
                            'owner_unit': weapon_data.get('owner_unit'),
                            'ammunition_count': weapon_data.get('ammunition_count', 0),
                            'status': weapon_data.get('status', 'ready'),
                            'last_fired': weapon_data.get('last_fired', 0),
                            'target': weapon_data.get('target'),
                            'last_update': time.time()
                        }
                logger.info(f"解析全量武器数据: {len(state.weapons)} 个武器")
                return True
        except Exception as e:
            logger.error(f"解析武器全量数据失败: {e}")
        return False

    def parse_update_frame(self, data: Dict[str, Any], state: SimulationState) -> bool:
        try:
            updated = False

            # 处理武器开火
            if 'weapon_fire_events' in data:
                for fire_event in data['weapon_fire_events']:
                    weapon_id = fire_event.get('weapon_id')
                    if weapon_id in state.weapons:
                        weapon = state.weapons[weapon_id]
                        weapon['last_fired'] = fire_event.get('timestamp', time.time())
                        weapon['ammunition_count'] = max(0, weapon['ammunition_count'] - 1)
                        weapon['status'] = 'firing'
                        weapon['target'] = fire_event.get('target')

                        # 添加开火事件到事件列表
                        event = {
                            'type': 'weapon_fire',
                            'weapon_id': weapon_id,
                            'timestamp': fire_event.get('timestamp'),
                            'target': fire_event.get('target'),
                            'projectile_type': fire_event.get('projectile_type')
                        }
                        state.events.append(event)
                        updated = True

            # 处理武器被摧毁
            if 'weapon_destroyed_events' in data:
                for destroy_event in data['weapon_destroyed_events']:
                    weapon_id = destroy_event.get('weapon_id')
                    if weapon_id in state.weapons:
                        state.weapons[weapon_id]['status'] = 'destroyed'
                        state.weapons[weapon_id]['destroyed_time'] = destroy_event.get('timestamp', time.time())

                        # 添加摧毁事件
                        event = {
                            'type': 'weapon_destroyed',
                            'weapon_id': weapon_id,
                            'timestamp': destroy_event.get('timestamp'),
                            'cause': destroy_event.get('cause', 'unknown')
                        }
                        state.events.append(event)
                        updated = True

            return updated
        except Exception as e:
            logger.error(f"解析武器更新数据失败: {e}")
        return False

    def get_priority(self) -> Priority:
        return Priority.HIGH


class EventParser(DataParser):
    """事件解析器"""

    def can_handle(self, msg_type: MessageType) -> bool:
        return msg_type in [MessageType.EVENT, MessageType.FULL_FRAME, MessageType.UPDATE_FRAME]

    def parse_full_frame(self, data: Dict[str, Any], state: SimulationState) -> bool:
        try:
            if 'events' in data:
                state.events.clear()
                state.events.extend(data['events'])
                logger.info(f"解析全量事件数据: {len(state.events)} 个事件")
                return True
        except Exception as e:
            logger.error(f"解析事件全量数据失败: {e}")
        return False

    def parse_update_frame(self, data: Dict[str, Any], state: SimulationState) -> bool:
        try:
            updated = False
            if 'new_events' in data:
                for event in data['new_events']:
                    event['received_time'] = time.time()
                    state.events.append(event)
                    updated = True

                # 限制事件列表长度，保留最近的1000个事件
                if len(state.events) > 1000:
                    state.events = state.events[-1000:]

            return updated
        except Exception as e:
            logger.error(f"解析事件更新数据失败: {e}")
        return False

    def get_priority(self) -> Priority:
        return Priority.NORMAL


class LogParser(DataParser):
    """日志解析器"""

    def can_handle(self, msg_type: MessageType) -> bool:
        return msg_type in [MessageType.LOG, MessageType.FULL_FRAME, MessageType.UPDATE_FRAME]

    def parse_full_frame(self, data: Dict[str, Any], state: SimulationState) -> bool:
        try:
            if 'logs' in data:
                state.logs.clear()
                state.logs.extend(data['logs'])
                logger.info(f"解析全量日志数据: {len(state.logs)} 条日志")
                return True
        except Exception as e:
            logger.error(f"解析日志全量数据失败: {e}")
        return False

    def parse_update_frame(self, data: Dict[str, Any], state: SimulationState) -> bool:
        try:
            updated = False
            if 'new_logs' in data:
                for log_entry in data['new_logs']:
                    log_entry['received_time'] = time.time()
                    state.logs.append(log_entry)
                    updated = True

                # 限制日志列表长度，保留最近的5000条日志
                if len(state.logs) > 5000:
                    state.logs = state.logs[-5000:]

            return updated
        except Exception as e:
            logger.error(f"解析日志更新数据失败: {e}")
        return False

    def get_priority(self) -> Priority:
        return Priority.LOW


class CommunicationParser(DataParser):
    """通信解析器"""

    def can_handle(self, msg_type: MessageType) -> bool:
        return msg_type in [MessageType.COMMUNICATION, MessageType.FULL_FRAME, MessageType.UPDATE_FRAME]

    def parse_full_frame(self, data: Dict[str, Any], state: SimulationState) -> bool:
        try:
            if 'communications' in data:
                state.communications.clear()
                state.communications.extend(data['communications'])
                logger.info(f"解析全量通信数据: {len(state.communications)} 条通信")
                return True
        except Exception as e:
            logger.error(f"解析通信全量数据失败: {e}")
        return False

    def parse_update_frame(self, data: Dict[str, Any], state: SimulationState) -> bool:
        try:
            updated = False
            if 'new_communications' in data:
                for comm in data['new_communications']:
                    comm['received_time'] = time.time()
                    state.communications.append(comm)
                    updated = True

                # 限制通信记录长度
                if len(state.communications) > 1000:
                    state.communications = state.communications[-1000:]

            return updated
        except Exception as e:
            logger.error(f"解析通信更新数据失败: {e}")
        return False

    def get_priority(self) -> Priority:
        return Priority.NORMAL


class SimulationDataManager:
    """仿真数据管理器 - 核心类"""

    def __init__(self):
        self.state = SimulationState()
        self.parsers: List[DataParser] = []
        self.message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.lock = threading.RLock()
        self.is_initialized = False
        self.processed_messages = set()
        self.max_message_history = 10000

        # 注册默认解析器
        self._register_default_parsers()

    def _register_default_parsers(self):
        """注册默认解析器"""
        self.parsers.extend([
            UnitStatusParser(),
            WeaponParser(),
            EventParser(),
            LogParser(),
            CommunicationParser()
        ])

        # 按优先级排序
        self.parsers.sort(key=lambda p: p.get_priority().value, reverse=True)

    def register_parser(self, parser: DataParser):
        """注册自定义解析器"""
        with self.lock:
            self.parsers.append(parser)
            self.parsers.sort(key=lambda p: p.get_priority().value, reverse=True)

    def register_message_handler(self, msg_type: MessageType, handler: Callable):
        """注册消息处理器"""
        self.message_handlers[msg_type].append(handler)

    def process_message(self, message: SimulationMessage) -> bool:
        """处理仿真消息"""
        with self.lock:
            # 去重检查
            if message.message_id in self.processed_messages:
                logger.debug(f"重复消息，已忽略: {message.message_id}")
                return False

            # 记录已处理消息
            self.processed_messages.add(message.message_id)
            if len(self.processed_messages) > self.max_message_history:
                # 清理老旧记录
                old_messages = list(self.processed_messages)[:len(self.processed_messages) // 2]
                for old_msg in old_messages:
                    self.processed_messages.discard(old_msg)

            success = False

            try:
                # 更新基本状态信息
                if message.game_id != self.state.game_id:
                    logger.info(f"检测到新游戏: {message.game_id}")
                    self.state.game_id = message.game_id
                    self.is_initialized = False

                self.state.timestamp = max(self.state.timestamp, message.timestamp)
                self.state.sequence_id = max(self.state.sequence_id, message.sequence_id)

                # 根据消息类型选择处理方式
                if message.msg_type == MessageType.FULL_FRAME:
                    success = self._process_full_frame(message)
                    if success:
                        self.is_initialized = True
                        logger.info("全量数据初始化完成")
                elif message.msg_type == MessageType.UPDATE_FRAME:
                    if self.is_initialized:
                        success = self._process_update_frame(message)
                    else:
                        logger.warning("尚未初始化，忽略增量更新")
                else:
                    # 处理特定类型消息
                    success = self._process_specific_message(message)

                # 触发消息处理器
                if success:
                    self._trigger_message_handlers(message)
                    self._update_statistics()

            except Exception as e:
                logger.error(f"处理消息时发生错误: {e}")
                success = False

            return success

    def _process_full_frame(self, message: SimulationMessage) -> bool:
        """处理全量帧"""
        success_count = 0
        for parser in self.parsers:
            if parser.can_handle(message.msg_type):
                try:
                    if parser.parse_full_frame(message.data, self.state):
                        success_count += 1
                except Exception as e:
                    logger.error(f"解析器 {parser.__class__.__name__} 处理全量帧失败: {e}")

        return success_count > 0

    def _process_update_frame(self, message: SimulationMessage) -> bool:
        """处理增量更新帧"""
        success_count = 0
        for parser in self.parsers:
            if parser.can_handle(message.msg_type):
                try:
                    if parser.parse_update_frame(message.data, self.state):
                        success_count += 1
                except Exception as e:
                    logger.error(f"解析器 {parser.__class__.__name__} 处理更新帧失败: {e}")

        return success_count > 0

    def _process_specific_message(self, message: SimulationMessage) -> bool:
        """处理特定类型消息"""
        success_count = 0
        for parser in self.parsers:
            if parser.can_handle(message.msg_type):
                try:
                    # 对于特定消息类型，使用update_frame方法处理
                    if parser.parse_update_frame(message.data, self.state):
                        success_count += 1
                except Exception as e:
                    logger.error(f"解析器 {parser.__class__.__name__} 处理特定消息失败: {e}")

        return success_count > 0

    def _trigger_message_handlers(self, message: SimulationMessage):
        """触发消息处理器"""
        for handler in self.message_handlers[message.msg_type]:
            try:
                handler(message, self.state)
            except Exception as e:
                logger.error(f"消息处理器执行失败: {e}")

    def _update_statistics(self):
        """更新统计信息"""
        self.state.stats = {
            'total_units': len(self.state.units),
            'active_units': len([u for u in self.state.units.values() if u.get('status') == 'active']),
            'total_weapons': len(self.state.weapons),
            'ready_weapons': len([w for w in self.state.weapons.values() if w.get('status') == 'ready']),
            'total_events': len(self.state.events),
            'total_logs': len(self.state.logs),
            'total_communications': len(self.state.communications),
            'last_update': time.time()
        }

    def get_state_snapshot(self) -> Dict[str, Any]:
        """获取当前状态快照"""
        with self.lock:
            return {
                'game_id': self.state.game_id,
                'timestamp': self.state.timestamp,
                'sequence_id': self.state.sequence_id,
                'is_initialized': self.is_initialized,
                'units': dict(self.state.units),
                'weapons': dict(self.state.weapons),
                'events': list(self.state.events),
                'logs': list(self.state.logs),
                'communications': list(self.state.communications),
                'stats': dict(self.state.stats)
            }

    def get_units_by_type(self, unit_type: str) -> List[Dict]:
        """根据类型获取单位"""
        with self.lock:
            return [unit for unit in self.state.units.values() if unit.get('type') == unit_type]

    def get_recent_events(self, limit: int = 100) -> List[Dict]:
        """获取最近的事件"""
        with self.lock:
            return self.state.events[-limit:] if self.state.events else []

    def get_unit_by_id(self, unit_id: str) -> Optional[Dict]:
        """根据ID获取单位"""
        with self.lock:
            return self.state.units.get(unit_id)


# 使用示例
def demo_usage():
    """演示如何使用框架"""

    # 创建数据管理器
    manager = SimulationDataManager()

    # 注册自定义消息处理器
    def on_weapon_fire(message: SimulationMessage, state: SimulationState):
        print(f"武器开火事件: {message.data}")

    def on_unit_destroyed(message: SimulationMessage, state: SimulationState):
        print(f"单位被摧毁: {message.data}")

    manager.register_message_handler(MessageType.WEAPON_FIRE, on_weapon_fire)
    manager.register_message_handler(MessageType.WEAPON_DESTROYED, on_unit_destroyed)

    # 模拟处理全量帧
    full_frame_data = {
        'units': [
            {
                'id': 'TANK_001',
                'type': 'main_battle_tank',
                'position': {'x': 100, 'y': 200, 'z': 0},
                'status': 'active',
                'health': 100,
                'fuel': 90
            },
            {
                'id': 'IFV_001',
                'type': 'infantry_fighting_vehicle',
                'position': {'x': 150, 'y': 180, 'z': 0},
                'status': 'active',
                'health': 85,
                'fuel': 95
            }
        ],
        'weapons': [
            {
                'id': 'CANNON_001',
                'type': '125mm_cannon',
                'owner_unit': 'TANK_001',
                'ammunition_count': 40,
                'status': 'ready'
            }
        ]
    }

    full_msg = SimulationMessage(
        msg_type=MessageType.FULL_FRAME,
        timestamp=time.time(),
        game_id="GAME_2024_001",
        sequence_id=1,
        data=full_frame_data
    )

    success = manager.process_message(full_msg)
    print(f"全量帧处理结果: {success}")

    # 模拟处理增量更新
    update_data = {
        'unit_updates': [
            {
                'id': 'TANK_001',
                'position': {'x': 105, 'y': 205},
                'fuel': 89
            }
        ],
        'weapon_fire_events': [
            {
                'weapon_id': 'CANNON_001',
                'timestamp': time.time(),
                'target': {'x': 300, 'y': 400},
                'projectile_type': 'APFSDS'
            }
        ]
    }

    update_msg = SimulationMessage(
        msg_type=MessageType.UPDATE_FRAME,
        timestamp=time.time(),
        game_id="GAME_2024_001",
        sequence_id=2,
        data=update_data
    )

    success = manager.process_message(update_msg)
    print(f"增量更新处理结果: {success}")

    # 获取状态快照
    snapshot = manager.get_state_snapshot()
    print(f"当前统计: {snapshot['stats']}")

    return manager


if __name__ == "__main__":
    demo_usage()