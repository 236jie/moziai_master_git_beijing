"""
训练回调：在每轮训练结束时保存最好的模型
"""
import os
import json
from typing import Dict, Any
from ray.tune import Callback
from mozi_ai_sdk.sdfk_test.model_manager import ModelManager


class ModelSaveCallback(Callback):
    """自定义回调：在每轮训练结束时保存最好的模型"""
    
    def __init__(self, model_manager: ModelManager):
        """
        初始化回调
        
        Args:
            model_manager: 模型管理器实例
        """
        super().__init__()
        self.model_manager = model_manager
        self.episode_data = {}  # 存储每轮训练的episode数据
    
    def on_trial_result(self, iteration: int, trials, trial, result: Dict[str, Any], **info):
        """
        在每次训练迭代结果时调用
        
        Args:
            iteration: 当前迭代次数
            trials: 所有trial列表
            trial: 当前trial
            result: 训练结果字典
        """
        # 从result中提取episode信息
        # Ray RLlib会将环境的extra_info中的信息聚合到custom_metrics中
        custom_metrics = result.get('custom_metrics', {})
        
        # 获取protected_facilities和missile_consumption
        protected_facilities = custom_metrics.get('protected_facilities_mean', 0)
        missile_consumption = custom_metrics.get('missile_consumption_mean', {})
        missile_cost = custom_metrics.get('missile_cost_mean', 0.0)
        
        # 如果missile_consumption是字典，直接使用；如果是列表，需要处理
        if isinstance(missile_consumption, list):
            # 如果是从多个episode聚合的，取最后一个或平均值
            if len(missile_consumption) > 0:
                missile_consumption = missile_consumption[-1] if isinstance(missile_consumption[-1], dict) else {}
            else:
                missile_consumption = {}
        
        # 如果missile_consumption为空，尝试从info中获取
        if not missile_consumption:
            info_dict = result.get('info', {})
            if isinstance(info_dict, dict):
                learner_info = info_dict.get('learner', {})
                # 尝试从其他地方获取
                pass
        
        # 存储当前迭代的数据
        iteration_key = f"iter_{iteration}"
        self.episode_data[iteration_key] = {
            'protected_facilities': protected_facilities,
            'missile_consumption': missile_consumption,
            'missile_cost': missile_cost,
            'iteration': iteration
        }
        
        # 获取当前检查点路径
        checkpoint_path = None
        if hasattr(trial, 'checkpoint') and trial.checkpoint:
            checkpoint_path = trial.checkpoint.dir_or_data
        elif hasattr(trial, 'checkpoint_dir') and trial.checkpoint_dir:
            # 查找最新的checkpoint
            checkpoint_dirs = [d for d in os.listdir(trial.checkpoint_dir) 
                              if d.startswith('checkpoint_')]
            if checkpoint_dirs:
                latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('_')[1]))
                checkpoint_path = os.path.join(trial.checkpoint_dir, latest_checkpoint)
        
        # 如果找到了检查点路径，尝试保存模型
        if checkpoint_path and os.path.exists(checkpoint_path):
            # 确保missile_consumption是字典格式
            if not isinstance(missile_consumption, dict):
                missile_consumption = {'C-400': 0, 'HQ-9A': 0, 'HQ-12': 0}
            
            # 尝试保存模型
            self.model_manager.try_save_model(
                protected_count=int(protected_facilities),
                missile_consumption=missile_consumption,
                iteration=iteration,
                checkpoint_path=checkpoint_path
            )
    
    def on_trial_complete(self, iteration: int, trials, trial):
        """在trial完成时调用"""
        pass

