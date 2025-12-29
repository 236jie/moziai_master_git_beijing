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


def test_do_nothing_flag_skips_actions():
    env = object.__new__(SDFKEnv)
    env.max_concurrent_targets = 24
    env.defense_types = ['C-400', 'HQ-9A', 'HQ-12']
    # 构造一个动作，前72位有一些非零，但最后一位表示 do-nothing
    action = [0] * (env.max_concurrent_targets * len(env.defense_types))
    action[0] = 3  # 如果不使用 do-nothing 标志，这会触发动作
    action.append(1)  # do-nothing 标志设置为 1

    engagement_plan = SDFKEnv._parse_action(env, action)
    assert engagement_plan == []
