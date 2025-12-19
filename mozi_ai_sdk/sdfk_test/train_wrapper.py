"""
训练包装器：集成监控、检查点管理、实时绘图
"""
import os
import json
import time
import threading
from collections import deque
import matplotlib.pyplot as plt
from mozi_ai_sdk.sdfk_test.training_monitor import TrainingMonitor, CheckpointManager


class SimpleTrainingMonitor:
    """简化的训练监控器（用于在训练循环外监控）"""
    
    def __init__(self, result_dir, save_dir="./training_logs", checkpoint_dir="./checkpoints"):
        self.result_dir = result_dir
        self.save_dir = save_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.best_results_file = os.path.join(save_dir, "best_results.json")
        self.best_results = []  # (reward, iteration, info) 列表
        
        # 加载历史
        self._load_best_results()
        
    def _load_best_results(self):
        if os.path.exists(self.best_results_file):
            try:
                with open(self.best_results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.best_results = [(item['reward'], item['iteration'], item['info']) for item in data]
            except:
                self.best_results = []
    
    def update_from_trial_result(self, iteration, result_dict):
        """从trial结果更新"""
        episode_reward_mean = result_dict.get('episode_reward_mean', 0)
        
        # 提取信息
        info = {
            'protected_facilities': result_dict.get('custom_metrics', {}).get('protected_facilities_mean', 0),
            'intercepted_missiles': result_dict.get('custom_metrics', {}).get('intercepted_missiles_mean', 0),
        }
        
        # 更新最好的10个结果
        self.best_results.append((episode_reward_mean, iteration, info))
        self.best_results.sort(key=lambda x: x[0], reverse=True)
        self.best_results = self.best_results[:10]
        
        # 保存
        data = [
            {
                'rank': i + 1,
                'reward': reward,
                'iteration': iteration,
                'info': info
            }
            for i, (reward, iteration, info) in enumerate(self.best_results)
        ]
        with open(self.best_results_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # 输出训练轮数
        print("=" * 100)
        print(f"训练轮数: {iteration} | 平均奖励: {episode_reward_mean:.2f} | "
              f"保护目标: {info.get('protected_facilities', 0):.1f} | "
              f"拦截数量: {info.get('intercepted_missiles', 0):.1f}")
        print("=" * 100)


def wrap_tune_run_with_monitoring(tune_run_func, *args, monitor=None, **kwargs):
    """包装tune.run函数以添加监控"""
    # 这里需要在tune.run执行时拦截结果
    # 由于Ray Tune的架构，我们采用不同的方法
    pass
