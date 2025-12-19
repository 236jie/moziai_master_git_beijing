"""
自定义Trainable类：包装Ray Tune训练以添加监控功能
"""
import os
import json
from ray.tune import Trainable
from ray.rllib.agents.ppo import PPOTrainer


class MonitoredTrainable(Trainable):
    """带监控功能的Trainable包装器"""
    
    def setup(self, config):
        """初始化训练"""
        # 创建PPO trainer
        self.trainer = PPOTrainer(config=config)
        
        # 监控相关
        self.best_results_file = os.path.join("./training_logs", "best_results.json")
        self.checkpoint_info_file = os.path.join("./checkpoints", "checkpoint_info.json")
        os.makedirs("./training_logs", exist_ok=True)
        os.makedirs("./checkpoints", exist_ok=True)
        
        self.best_results = []  # (reward, iteration) 列表
        self.best_checkpoints = []  # (reward, checkpoint_path, iteration) 列表
        self._load_history()
    
    def _load_history(self):
        """加载历史记录"""
        if os.path.exists(self.best_results_file):
            try:
                with open(self.best_results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.best_results = [(item['reward'], item['iteration']) for item in data]
            except:
                self.best_results = []
        
        if os.path.exists(self.checkpoint_info_file):
            try:
                with open(self.checkpoint_info_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.best_checkpoints = [(item['reward'], item['checkpoint_path'], item['iteration']) 
                                            for item in data]
            except:
                self.best_checkpoints = []
    
    def _save_results(self):
        """保存最好的结果"""
        if len(self.best_results) > 0:
            sorted_results = sorted(self.best_results, key=lambda x: x[0], reverse=True)[:10]
            data = [
                {'rank': i+1, 'reward': r, 'iteration': it}
                for i, (r, it) in enumerate(sorted_results)
            ]
            with open(self.best_results_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    def step(self):
        """执行一步训练"""
        result = self.trainer.train()
        
        # 提取信息
        iteration = self.iteration
        reward = result.get('episode_reward_mean', 0)
        
        # 更新最好的结果
        self.best_results.append((reward, iteration))
        self.best_results.sort(key=lambda x: x[0], reverse=True)
        self.best_results = self.best_results[:10]
        self._save_results()
        
        # 输出训练轮数（用分隔线）
        print("=" * 100)
        print(f"训练轮数: {iteration} | 平均奖励: {reward:.2f}")
        print("=" * 100)
        
        return result
    
    def save_checkpoint(self, checkpoint_dir):
        """保存检查点"""
        checkpoint_path = self.trainer.save(checkpoint_dir)
        
        # 更新最好的检查点（只保留5个）
        reward = self.trainer.get_policy().get_last_mean_reward() if hasattr(self.trainer.get_policy(), 'get_last_mean_reward') else 0
        self.best_checkpoints.append((reward, checkpoint_path, self.iteration))
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)
        self.best_checkpoints = self.best_checkpoints[:5]
        
        # 保存检查点信息
        data = [
            {'rank': i+1, 'reward': r, 'checkpoint_path': p, 'iteration': it}
            for i, (r, p, it) in enumerate(self.best_checkpoints)
        ]
        with open(self.checkpoint_info_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        self.trainer.restore(checkpoint_path)
    
    def cleanup(self):
        """清理资源"""
        self.trainer.stop()
