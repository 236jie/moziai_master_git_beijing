"""
改进建议代码示例
本文件展示了如何修复env_sdfk.py中的关键问题
"""

# ==================== 1. 在 __init__ 方法中添加的变量 ====================

def __init__improved__(self, env_config):
    # ... 原有代码 ...
    
    # ===== 新增：追踪蓝方导弹状态，用于检测拦截成功 =====
    self.detected_blue_missiles = {}  # {target_id: missile_obj} - 当前探测到的蓝方导弹
    self.intercepted_missiles = set()  # 已成功拦截的导弹ID集合
    self.missile_interception_history = {}  # {target_id: [拦截步数列表]} - 记录哪些步对该目标进行了拦截
    
    # 拦截成功奖励配置（可通过env_config配置）
    self.intercept_success_reward = env_config.get('intercept_success_reward', 5.0)  # 拦截成功奖励
    self.intercept_failure_penalty = env_config.get('intercept_failure_penalty', 1.0)  # 拦截失败惩罚（目标接近但未被拦截）


# ==================== 2. 新增方法：检测拦截成功 ====================

def _check_interception_success(self):
    """
    检测拦截成功的导弹
    
    原理：
    1. 对比上一 step 和当前 step 探测到的导弹列表
    2. 如果某个导弹消失了，且最近几步对它进行过拦截，则认为是拦截成功
    3. 给予拦截成功奖励
    
    Returns:
        int: 本次检测到的拦截成功数量
    """
    # 获取当前探测到的所有蓝方导弹
    current_detected = {k: v for k, v in self.side.contacts.items() 
                       if v.m_ContactType == 1}  # m_ContactType == 1 表示导弹
    
    current_ids = set(current_detected.keys())
    previous_ids = set(self.detected_blue_missiles.keys())
    
    # 找出消失的导弹（可能被拦截了）
    disappeared_ids = previous_ids - current_ids
    
    interception_count = 0
    for missile_id in disappeared_ids:
        # 检查该导弹是否在最近几步被拦截过
        if missile_id in self.missile_interception_history:
            # 检查最近几步（例如最近10步）是否进行过拦截
            recent_engagement_steps = [
                step for step in self.missile_interception_history[missile_id]
                if (self.steps - step) <= 10
            ]
            
            if recent_engagement_steps:
                # 认为是拦截成功
                if missile_id not in self.intercepted_missiles:
                    self.intercepted_missiles.add(missile_id)
                    self.temp_reward += self.intercept_success_reward
                    interception_count += 1
                    print(f"[步骤{self.steps}] 拦截成功！导弹 {missile_id} 被击落，奖励 +{self.intercept_success_reward}")
        
        # 清理该导弹的历史记录
        if missile_id in self.missile_interception_history:
            del self.missile_interception_history[missile_id]
    
    # 更新当前探测到的导弹列表
    self.detected_blue_missiles = current_detected
    
    return interception_count


def _check_interception_failure(self):
    """
    检测拦截失败（目标接近保护目标但未被拦截）
    
    对于距离保护目标很近（<20km）但仍未被拦截的导弹，给予小惩罚
    这样可以鼓励尽早拦截，避免等到最后一刻
    """
    failure_count = 0
    for missile_id, missile in self.detected_blue_missiles.items():
        if missile_id in self.intercepted_missiles:
            continue  # 已拦截，跳过
        
        distance = self._get_distance_to_protected_targets(missile)
        # 如果距离很近（<20km）且没有被拦截，说明拦截可能失败
        if distance < 20:
            failure_count += 1
    
    if failure_count > 0:
        penalty = failure_count * self.intercept_failure_penalty
        self.temp_reward -= penalty
        if self.steps % 10 == 0:
            print(f"[步骤{self.steps}] 检测到 {failure_count} 个目标接近保护区域且未被拦截，惩罚 -{penalty}")
    
    return failure_count


# ==================== 3. 修改 _execute_engagement_plan 方法 ====================

def _execute_engagement_plan_improved(self, engagement_plan):
    """
    执行拦截计划（改进版）
    在执行拦截时，记录拦截历史
    """
    executed_engagements = []

    for plan in engagement_plan:
        target_id = plan['target_id']
        target = plan['target']
        assignments = plan['defense_assignments']
        target_distance = plan['target_distance']

        # ===== 新增：记录拦截历史 =====
        if target_id not in self.missile_interception_history:
            self.missile_interception_history[target_id] = []
        
        # 如果有任何拦截动作，记录本次拦截
        if any(count > 0 for count in assignments.values()):
            self.missile_interception_history[target_id].append(self.steps)
        # ===== 新增代码结束 =====

        # ... 原有拦截逻辑 ...
        # （省略，保持原有代码不变）

        # ... 原有代码 ...

    return executed_engagements


# ==================== 4. 修改 step 方法 ====================

def step_improved(self, action_dict):
    """
    RL标准接口（改进版）
    在执行动作后、更新态势前，先检测拦截成功
    """
    done = False
    action = action_dict['agent_0']
    engagement_plan = self._parse_action(action)

    if self.env_config['mode'] in ['train', 'development']:
        force_done = self.safe_step(action, engagement_plan)
        if force_done:
            done = force_done
            self.reset_nums = 4
            print(f"{time.strftime('%H:%M:%S')} 在第{self.steps}步，强制重启墨子！！！")
        else:
            # ===== 新增：在更新态势前检测拦截成功 =====
            interception_count = self._check_interception_success()
            failure_count = self._check_interception_failure()
            # ===== 新增代码结束 =====
            
            self._update(self.scenario)
            done = self._is_done()
    elif self.env_config['mode'] in ['versus', 'eval']:
        self._execute_engagement_plan(engagement_plan)
        self.scenario = self.env.step()
        
        # ===== 新增：检测拦截成功 =====
        interception_count = self._check_interception_success()
        failure_count = self._check_interception_failure()
        # ===== 新增代码结束 =====
        
        self._update(self.scenario)
        done = self._is_done()
    
    reward = {'agent_0': self.reward}
    obs_array = np.array(self._generate_features(), dtype=np.float32)
    obs = {'agent_0': obs_array}
    self.steps += 1
    
    if self.steps % 10 == 0:
        print(f'第 {self.steps}步' + '-' + f'reward is {self.reward}')
    
    if done:
        print('++++Score:', self.reward_accum, 'step:', self.steps)
        print(f'本局拦截成功数量: {len(self.intercepted_missiles)}')
    
    return obs, reward, {'__all__': done, 'agent_0': done}, {'agent_0': {'score': self._get_win_score()}}


# ==================== 5. 修改 reset 方法 ====================

def reset_improved(self):
    """
    RL标准接口（改进版）
    重置时清理拦截追踪变量
    """
    self._get_initial_state()
    self.steps = 0
    self.temp_reward = 0
    
    # ===== 新增：清理拦截追踪变量 =====
    self.detected_blue_missiles = {}
    self.intercepted_missiles = set()
    self.missile_interception_history = {}
    # ===== 新增代码结束 =====
    
    self.recent_engagements = []
    self.reward_accum = self._get_win_score()
    self._update(self.scenario)
    self.reward = 0.0
    obs = {'agent_0': np.array(self._generate_features(), dtype=np.float32)}
    print('env_reset finished!!!')
    return obs


# ==================== 6. 改进奖励函数 ====================

def _get_win_score_improved(self):
    """
    计算当前局势的得分（改进版）
    
    改进点：
    1. 拦截成功奖励已在 _check_interception_success 中通过 temp_reward 添加
    2. 增大保护目标损失的惩罚，使信号更明显
    3. 缩小存活奖励，避免信号过密
    4. 添加拦截失败惩罚（通过 _check_interception_failure）
    """
    score = 0.0
    
    # 1. 计算保护目标损失惩罚（增大惩罚力度）
    self.protected_target = {k: v.strName for k, v in self.side.facilities.items()
                             if v.m_Category == 3001 and '雷达' not in v.strName}
    lost_count = len(self.init_protected_facility) - len(self.protected_target)
    score -= lost_count * 50.0  # 从20.0增大到50.0，使信号更明显
    
    # 2. 添加保护目标存活的奖励（缩小奖励，避免信号过密）
    detected_missiles = {k: v for k, v in self.side.contacts.items() 
                        if v.m_ContactType == 1}
    has_threat = len(detected_missiles) > 0
    
    if has_threat:
        remaining_count = len(self.protected_target)
        score += remaining_count * 0.1  # 从0.5降低到0.1，避免奖励过密
    
    # 3. 若所有保护目标被摧毁，叠加一次性终局大惩罚
    if len(self.protected_target) == 0 and lost_count > 0:
        score -= 500.0  # 从200.0增大到500.0
    
    # 4. 拦截统计奖励（新增：鼓励拦截更多目标）
    intercept_rate = len(self.intercepted_missiles) / max(len(detected_missiles) + len(self.intercepted_missiles), 1)
    score += intercept_rate * 2.0  # 拦截率奖励，最高2.0分
    
    if self.steps % 10 == 0:
        print(f'红方地面核心设施还剩下{len(self.protected_target)}个，'
              f'已拦截{len(self.intercepted_missiles)}个目标，得分{score/100:.2f}')
    
    return float(score) / 100


# ==================== 7. 改进观测空间（可选） ====================

def _generate_features_improved(self):
    """
    构造观测特征向量（改进版）
    
    改进点：
    1. 增加最危险目标的详细信息（而不是使用平均值）
    2. 增加拦截统计信息
    3. 增加目标数量变化信息
    """
    feats = []
    
    # 1. 最危险的3个目标详细信息（3×7=21维）
    detected_missiles = {k: v for k, v in self.side.contacts.items() 
                        if v.m_ContactType == 1}
    
    # 计算威胁度并排序
    threats = []
    for missile_id, missile in detected_missiles.items():
        threat_score = self._calculate_threat_score(missile)
        distance = self._get_distance_to_protected_targets(missile)
        speed = missile.fCurrentSpeed if hasattr(missile, 'fCurrentSpeed') else 0
        heading = missile.fCurrentHeading if hasattr(missile, 'fCurrentHeading') else 0
        eta = distance / speed if speed > 0 else 999
        
        threats.append({
            'id': missile_id,
            'distance': distance,
            'speed': speed,
            'heading': heading,
            'eta': eta,
            'threat_score': threat_score,
            'in_range_c400': 1.0 if distance <= RED_MISSILE_RANGE['C-400'] else 0.0,
            'in_range_hq9': 1.0 if distance <= RED_MISSILE_RANGE['HQ-9A'] else 0.0,
            'in_range_hq12': 1.0 if distance <= RED_MISSILE_RANGE['HQ-12'] else 0.0,
        })
    
    threats.sort(key=lambda x: x['threat_score'], reverse=True)
    top_threats = threats[:3]
    
    # 填充最危险3个目标的特征
    for threat in top_threats:
        feats.extend([
            threat['distance'] / 200.0,  # 归一化距离
            threat['speed'] / 1000.0,    # 归一化速度
            threat['heading'] / 360.0,   # 归一化航向
            min(threat['eta'] / 1800.0, 1.0),  # 预计到达时间（秒）/30分钟，上限1.0
            threat['in_range_c400'],
            threat['in_range_hq9'],
            threat['in_range_hq12'],
        ])
    
    # 如果威胁不足3个，用0填充
    while len(feats) < 21:
        feats.extend([0.0] * 7)
    
    # 2. 目标统计信息（新增）
    feats.append(len(detected_missiles) / 24.0)  # 当前目标数量归一化
    feats.append(len(self.intercepted_missiles) / 100.0)  # 已拦截数量归一化
    intercept_rate = len(self.intercepted_missiles) / max(len(detected_missiles) + len(self.intercepted_missiles), 1)
    feats.append(intercept_rate)  # 拦截成功率
    
    # 3. 红方防空单元状态（保持原有逻辑，9维）
    c400_units = [v for k, v in self.side.facilities.items() if 'C-400' in v.strName]
    c400_feats = self._get_unit_status_feats(c400_units, 'C-400')
    feats.extend(c400_feats)
    
    hq9_units = [v for k, v in self.side.facilities.items() if 'HQ-9A' in v.strName]
    hq9_feats = self._get_unit_status_feats(hq9_units, 'HQ-9A')
    feats.extend(hq9_feats)
    
    hq12_units = [v for k, v in self.side.facilities.items() if 'HQ-12' in v.strName]
    hq12_feats = self._get_unit_status_feats(hq12_units, 'HQ-12')
    feats.extend(hq12_feats)
    
    # 4. 时间进度特征（保持原有逻辑，4维）
    time_delta = self.m_Time - self.m_StartTime
    feats.append(time_delta / 1800.0)
    feats.append(time_delta / 3600.0)
    feats.append(time_delta / 4500.0)
    feats.append(time_delta / 5400.0)
    
    # 总共：21 + 3 + 9 + 4 = 37维
    return feats


# ==================== 使用说明 ====================

"""
如何应用这些改进：

1. 在 env_sdfk.py 的 __init__ 方法中：
   - 添加 self.detected_blue_missiles = {}
   - 添加 self.intercepted_missiles = set()
   - 添加 self.missile_interception_history = {}

2. 在 env_sdfk.py 中添加新方法：
   - _check_interception_success()
   - _check_interception_failure()

3. 修改 step() 方法：
   - 在执行拦截后、调用 _update() 前，调用 _check_interception_success()
   - 可选：调用 _check_interception_failure()

4. 修改 _execute_engagement_plan() 方法：
   - 记录拦截历史到 self.missile_interception_history

5. 修改 reset() 方法：
   - 清理拦截追踪变量

6. 可选：改进 _get_win_score() 和 _generate_features()
"""
