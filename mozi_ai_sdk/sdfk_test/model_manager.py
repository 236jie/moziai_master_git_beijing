"""
模型管理器：基于红方地面核心设施剩余数量和导弹消耗费用保存最好的10个模型
"""
import os
import json
import shutil
import signal
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# 导弹价值配置（从env_sdfk.py导入）
RED_MISSILE_COST = {"C-400": 30, "HQ-9A": 20, "HQ-12": 10}


class ModelManager:
    """模型管理器：保存训练过程中最好的10个模型"""
    
    def __init__(self, save_dir="./best_models", top_k=10):
        """
        初始化模型管理器
        
        Args:
            save_dir: 模型保存目录
            top_k: 保存最好的K个模型
        """
        self.save_dir = save_dir
        self.top_k = top_k
        os.makedirs(save_dir, exist_ok=True)
        
        # 模型信息文件
        self.info_file = os.path.join(save_dir, "model_info.json")
        
        # 已保存的模型列表，格式: [(protected_count, missile_cost, iteration, checkpoint_path), ...]
        self.saved_models: List[Tuple[int, float, int, str]] = []
        
        # 加载已保存的模型信息
        self._load_saved_models()
        
        # 注册信号处理，确保强制终止时也能保存
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # 当前训练轮次的临时信息（用于强制终止时保存）
        self.current_iteration = None
        self.current_checkpoint_path = None
        self.current_protected_count = None
        self.current_missile_cost = None
    
    def _signal_handler(self, signum, frame):
        """信号处理函数：在强制终止时保存当前模型"""
        print(f"\n收到终止信号 ({signum})，正在保存当前模型...")
        if (self.current_iteration is not None and 
            self.current_checkpoint_path and 
            os.path.exists(self.current_checkpoint_path)):
            try:
                self._save_model(
                    protected_count=self.current_protected_count or 0,
                    missile_cost=self.current_missile_cost or float('inf'),
                    iteration=self.current_iteration,
                    checkpoint_path=self.current_checkpoint_path
                )
                print("当前模型已保存")
            except Exception as e:
                print(f"保存当前模型时出错: {e}")
        sys.exit(0)
    
    def _load_saved_models(self):
        """加载已保存的模型信息"""
        if os.path.exists(self.info_file):
            try:
                with open(self.info_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.saved_models = [
                        (item['protected_count'], item['missile_cost'], 
                         item['iteration'], item['checkpoint_path'])
                        for item in data
                    ]
                    print(f"已加载 {len(self.saved_models)} 个已保存的模型")
            except Exception as e:
                print(f"加载模型信息失败: {e}")
                self.saved_models = []
        else:
            self.saved_models = []
    
    def _save_model_info(self):
        """保存模型信息到JSON文件"""
        data = [
            {
                'rank': i + 1,
                'protected_count': protected_count,
                'missile_cost': missile_cost,
                'iteration': iteration,
                'checkpoint_path': checkpoint_path
            }
            for i, (protected_count, missile_cost, iteration, checkpoint_path) 
            in enumerate(self.saved_models)
        ]
        with open(self.info_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _calculate_missile_cost(self, missile_consumption: dict) -> float:
        """
        计算导弹消耗总费用
        
        Args:
            missile_consumption: 导弹消耗字典，格式 {"C-400": count, "HQ-9A": count, "HQ-12": count}
        
        Returns:
            总费用
        """
        total_cost = 0.0
        for missile_type, count in missile_consumption.items():
            cost_per_missile = RED_MISSILE_COST.get(missile_type, 0)
            total_cost += cost_per_missile * count
        return total_cost
    
    def _should_save(self, protected_count: int, missile_cost: float) -> bool:
        """
        判断是否应该保存模型
        
        Args:
            protected_count: 红方地面核心设施剩余数量
            missile_cost: 导弹消耗总费用
        
        Returns:
            是否应该保存
        """
        # 如果已保存的模型少于top_k个，直接保存
        if len(self.saved_models) < self.top_k:
            return True
        
        # 如果已保存了top_k个，检查是否比最差的更好
        # 找到最差的模型（protected_count最小，如果相同则missile_cost最大）
        worst_protected = min(m[0] for m in self.saved_models)
        worst_cost = max(m[1] for m in self.saved_models if m[0] == worst_protected)
        
        # 比较：优先比较protected_count，如果相同则比较missile_cost
        if protected_count > worst_protected:
            return True
        elif protected_count == worst_protected and missile_cost < worst_cost:
            return True
        
        return False
    
    def _save_model(self, protected_count: int, missile_cost: float, 
                    iteration: int, checkpoint_path: str) -> Optional[str]:
        """
        保存模型到指定目录
        
        Args:
            protected_count: 红方地面核心设施剩余数量
            missile_cost: 导弹消耗总费用
            iteration: 训练迭代次数
            checkpoint_path: Ray Tune生成的检查点路径
        
        Returns:
            保存后的模型路径，如果保存失败返回None
        """
        if not os.path.exists(checkpoint_path):
            print(f"警告：检查点路径不存在: {checkpoint_path}")
            return None
        
        # 创建保存目录
        model_dir = os.path.join(
            self.save_dir, 
            f"rank_{len(self.saved_models)+1}_iter_{iteration}_"
            f"protected_{protected_count}_cost_{missile_cost:.1f}"
        )
        
        # 如果目录已存在，先删除
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        
        try:
            # 复制检查点
            shutil.copytree(checkpoint_path, model_dir)
            print(f"✓ 保存模型: 迭代={iteration}, 保护目标={protected_count}, "
                  f"导弹费用={missile_cost:.1f}, 路径={model_dir}")
            return model_dir
        except Exception as e:
            print(f"保存模型失败: {e}")
            return None
    
    def update_current_training(self, iteration: int, checkpoint_path: str,
                                protected_count: int = None, missile_cost: float = None):
        """更新当前训练轮次的信息（用于强制终止时保存）"""
        self.current_iteration = iteration
        self.current_checkpoint_path = checkpoint_path
        self.current_protected_count = protected_count
        self.current_missile_cost = missile_cost
    
    def try_save_model(self, protected_count: int, missile_consumption: dict,
                      iteration: int, checkpoint_path: str) -> bool:
        """
        尝试保存模型（如果满足条件）
        
        Args:
            protected_count: 红方地面核心设施剩余数量
            missile_consumption: 导弹消耗字典
            iteration: 训练迭代次数
            checkpoint_path: Ray Tune生成的检查点路径
        
        Returns:
            是否成功保存
        """
        # 计算导弹消耗费用
        missile_cost = self._calculate_missile_cost(missile_consumption)
        
        # 更新当前训练信息
        self.update_current_training(iteration, checkpoint_path, protected_count, missile_cost)
        
        # 判断是否应该保存
        if not self._should_save(protected_count, missile_cost):
            return False
        
        # 保存模型
        saved_path = self._save_model(protected_count, missile_cost, iteration, checkpoint_path)
        if saved_path is None:
            return False
        
        # 更新已保存模型列表
        self.saved_models.append((protected_count, missile_cost, iteration, saved_path))
        
        # 如果超过top_k个，删除最差的
        if len(self.saved_models) > self.top_k:
            # 排序：优先按protected_count降序，相同则按missile_cost升序
            self.saved_models.sort(key=lambda x: (-x[0], x[1]))
            # 删除最差的
            worst = self.saved_models.pop()
            worst_path = worst[3]
            if os.path.exists(worst_path):
                try:
                    shutil.rmtree(worst_path)
                    print(f"删除最差模型: {worst_path}")
                except Exception as e:
                    print(f"删除最差模型失败: {e}")
        
        # 重新排序并更新排名
        self.saved_models.sort(key=lambda x: (-x[0], x[1]))
        # 重新命名目录以反映新排名
        temp_models = []
        for i, (protected_count, missile_cost, iteration, old_path) in enumerate(self.saved_models):
            new_dir = os.path.join(
                self.save_dir,
                f"rank_{i+1}_iter_{iteration}_"
                f"protected_{protected_count}_cost_{missile_cost:.1f}"
            )
            if old_path != new_dir and os.path.exists(old_path):
                if os.path.exists(new_dir):
                    shutil.rmtree(new_dir)
                os.rename(old_path, new_dir)
                temp_models.append((protected_count, missile_cost, iteration, new_dir))
            else:
                temp_models.append((protected_count, missile_cost, iteration, old_path))
        self.saved_models = temp_models
        
        # 保存模型信息
        self._save_model_info()
        
        return True

