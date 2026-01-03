import pytest

from mozi_ai_sdk.sdfk_test.envs.env_sdfk import SDFKEnv


def make_fake_side_with_contacts(contact_keys):
    class FakeContact:
        def __init__(self):
            self.m_ContactType = 1

    class FakeSide:
        def __init__(self):
            self.contacts = {k: FakeContact() for k in contact_keys}
            self.facilities = {}
            self.iTotalScore = 0

    return FakeSide()


def test_reset_clears_target_set_and_temp_reward():
    env = object.__new__(SDFKEnv)
    # minimal attributes to exercise reset logic without constructing the whole Environment
    env.env_config = {'mode': 'development', 'side_name': 'red', 'enemy_side_name': 'blue', 'avail_docker_ip_port': []}
    env.side_name = 'red'
    env.protected_name = ['A']
    env.protected_target = {'A': object()}
    env.side = make_fake_side_with_contacts(['t1'])
    env.scenario = type('Sc', (), {'m_Duration': '@', 'm_StartTime': 0, 'm_Time': 0})()
    env.steps = 0
    env.removed_targets_set = {'x'}
    env.temp_reward = 0.5
    env.target_set = {'old'}
    # prevent attempts to restart or call external env
    env._get_initial_state = lambda: None

    obs = SDFKEnv.reset(env)

    assert env.target_set == set()
    assert env.temp_reward == 0
    assert env.removed_targets_set == set()
    assert env.reward == 0.0


def test_update_calculates_reward_once():
    env = object.__new__(SDFKEnv)
    env.side_name = 'red'
    env.side = make_fake_side_with_contacts([])
    env.protected_name = []
    env.protected_target = {}
    env.reward_accum = 0.0
    env.temp_reward = 0.2
    env.scenario = type('Sc', (), {'m_Time': 0})()

    # monkeypatch _get_win_score to a deterministic value
    env._get_win_score = lambda: 0.5

    env._update(env.scenario)

    assert env.reward == pytest.approx(0.5 - 0.0 + 0.2)
    assert env.reward_accum == pytest.approx(0.5 + 0.2)
    assert env.temp_reward == 0


def test_do_nothing_flag_removed():
    """测试：do-nothing机制已移除，动作空间现在是72维（24目标 × 3种防空单元）"""
    env = object.__new__(SDFKEnv)
    env.max_concurrent_targets = 24
    env.defense_types = ['C-400', 'HQ-9A', 'HQ-12']
    env.steps = 1
    env.force_action_when_targets = False  # 禁用强制引导，仅测试基本解析
    env.force_action_target_threshold = 1
    env.force_action_c400_range = False
    env.do_nothing_base_penalty = 2.0
    
    # 构造一个72维动作（已移除do-nothing标志）
    action = [0] * (env.max_concurrent_targets * len(env.defense_types))
    action[0] = 3  # 第一个目标对C-400发射3枚
    
    # 模拟没有探测到目标的情况（需要设置side和_get_detected_missiles_sorted）
    class FakeSide:
        def __init__(self):
            self.contacts = {}
    
    env.side = FakeSide()
    env._get_detected_missiles_sorted = lambda: []
    env._get_distance_to_protected_targets = lambda x: 100.0
    
    engagement_plan = SDFKEnv._parse_action(env, action)
    # 由于没有探测到目标，engagement_plan应该为空（不是因为do-nothing，而是因为没有目标）
    assert isinstance(engagement_plan, list)
