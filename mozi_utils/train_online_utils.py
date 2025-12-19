# 时间 : 2022/08/26 17:41
# 作者 : 张志高
# 文件 : train_online_utils
# 项目 : 墨子联合作战智能体研发平台
# 版权 : 北京华戍防务技术有限公司


# 放到mozi_utils文件夹下，供智能体调用

from websocket import create_connection
import json


def _dict_key_to_upper(dict_message):
    """
    功能：将字典中的key转换成大写
    参数：
        dict_message，dict
    作者：张志高
    时间：2022-8-28
    返回：无
    """
    dict_message_new = {}
    for k, v in dict_message.items():
        dict_message_new[k.upper()] = v
    return dict_message_new


def _send_message(_backend_ip_port, _message):
    """
    功能：向网站后端发送信息
    参数：
        _backend_ip_port，str, 网站后端IP和通信端口，格式 192.168.1.44:4567
        _str_message，json格式信息
    作者：张志高
    时间：2022-8-26
    返回：无
    """
    uri = f"ws://{_backend_ip_port}"
    ws = create_connection(uri)
    message = _dict_key_to_upper(_message)
    print(f"发送信息给网站后端：{message}")
    ws.send(json.dumps(message, ensure_ascii=False))
    # result = ws.recv()
    # print(f"接收到反馈信息：{result}")
    ws.close()


def report_docker_ip_port(_backend_ip_port, _training_id, _agent_id, _docker_list):
    """
    功能：向网站后端上报启动容器列表
    参数：
        _backend_ip_port，str, 网站后端IP和通信端口，格式 192.168.1.44:4567
        _training_id，int, 训练ID
        _agent_id，int, 智能体ID
        _docker_list, 容器字典列表，示例：
        [
            {
                'container_name': 'mozi_ai_01',
                'server_ip': '192.168.1.44',
                'docker_port': 3005
            },
                        {
                'container_name': 'mozi_ai_02',
                'server_ip': '192.168.1.44',
                'docker_port': 3006
            }
        ]
    作者：张志高
    时间：2022-8-26
    返回：无
    """
    message = {
        'interface_code': 'TRAIN_02',
        'message_type': 'report_docker_list',
        'training_id': _training_id,
        'agent_id': _agent_id,
        'docker_list': _docker_list
    }
    _send_message(_backend_ip_port, message)


def report_game_number_change(_backend_ip_port, _training_id, _agent_id, _newly_added_game_number):
    """
    功能：向网站后端上报新增训练局数
    参数：
        _backend_ip_port，str, 网站后端IP和通信端口，格式 192.168.1.44:4567
        _training_id，int, 训练ID
        _agent_id，int, 智能体ID
        _newly_added_game_number，int, 新增训练局数
    作者：张志高
    时间：2022-8-26
    返回：无
    """
    message = {
        'interface_code': 'TRAIN_02',
        'message_type': 'report_newly_added_game_number',
        'training_id': _training_id,
        'agent_id': _agent_id,
        'newly_added_game_number': _newly_added_game_number
    }
    _send_message(_backend_ip_port, message)


def report_train_complete(_backend_ip_port, _training_id, _agent_id):
    """
    功能：向网站后端上报训练完成
    参数：
        _backend_ip_port，str, 网站后端IP和通信端口，格式 192.168.1.44:4567
        _training_id，int, 训练ID
        _agent_id，int, 智能体ID
    作者：张志高
    时间：2022-8-26
    返回：无
    """
    message_type = 'train_complete'
    if _training_id == -1:          # 训练ID为-1时，判定为训练测试
        message_type = 'train_test_complete'

    message = {
        'interface_code': 'TRAIN_02',
        'message_type': message_type,
        'training_id': _training_id,
        'agent_id': _agent_id
    }
    _send_message(_backend_ip_port, message)


def report_train_error(_backend_ip_port, _training_id, _agent_id, _error_message):
    """
    功能：向网站后端上报训练完成
    参数：
        _backend_ip_port，str, 网站后端IP和通信端口，格式 192.168.1.44:4567
        _training_id，int, 训练ID
        _agent_id，int, 智能体ID
        _error_message, str, 训练错误信息
    作者：张志高
    时间：2022-8-26
    返回：无
    """
    message_type = 'train_error'
    if _training_id == -1:          # 训练ID为-1时，判定为训练测试
        message_type = 'train_test_error'
    message = {
        'interface_code': 'TRAIN_02',
        'message_type': message_type,
        'training_id': _training_id,
        'agent_id': _agent_id,
        'error_message': _error_message
    }
    _send_message(_backend_ip_port, message)


if __name__ == '__main__':
    backend_ip_port = '127.0.0.1:8766'
    docker_list = [
        {
            'container_name': 'mozi_ai_01',
            'server_ip': '192.168.1.44',
            'docker_port': 3005
        },
        {
            'container_name': 'mozi_ai_02',
            'server_ip': '192.168.1.44',
            'docker_port': 3006
        }
    ]
    report_docker_ip_port(backend_ip_port, 123, 456, docker_list)
    report_game_number_change(backend_ip_port, 123, 456, 3)
    report_train_complete(backend_ip_port, 123, 456)
    report_train_error(backend_ip_port, 123, 456, 'error123')

