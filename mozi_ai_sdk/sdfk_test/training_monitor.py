"""
训练监控模块：记录最好的训练结果、保存检查点、实时绘图
"""
import os
import json
import heapq
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from threading import Lock
import time

class TrainingMonitor:
    """训练监控器：记录最好的结果、实时绘图"""
    
    def __init__(self, save_dir="./training_logs", top_k=10, checkpoint_dir="./checkpoints"):
        self.save_dir = save_dir
        self.checkpoint_dir = checkpoint_dir
        self.top_k = top_k
        
        # 创建目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 记录最好的K个结果（最小堆，存储最差的K个）
        self.best_results = []  # 存储 (reward, iteration, info) 的元组
        self.best_results_file = os.path.join(save_dir, "best_results.json")
        
        # 实时绘图数据
        self.plot_data = {
            'iterations': deque(maxlen=1000),  # 最多保存1000个点
            'rewards': deque(maxlen=1000),
            'protected_facilities': deque(maxlen=1000),
            'intercepted_missiles': deque(maxlen=1000),
        }
        
        # 绘图相关
        self.fig = None
        self.axes = None
        self.lock = Lock()  # 线程锁
        self.plot_update_interval = 10  # 每10次迭代更新一次图
        
        # 加载历史最好的结果
        self._load_best_results()
        
    def _load_best_results(self):
        """加载历史最好的结果"""
        if os.path.exists(self.best_results_file):
            try:
                with open(self.best_results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.best_results = [(item['reward'], item['iteration'], item['info']) 
                                        for item in data]
                    # 确保是最小堆（最差的是堆顶）
                    heapq.heapify(self.best_results)
            except Exception as e:
                print(f"加载历史最好结果失败: {e}")
                self.best_results = []
        else:
            self.best_results = []
    
    def _save_best_results(self):
        """保存最好的结果到文件"""
        with self.lock:
            # 转换为列表并排序（从最好到最差）
            sorted_results = sorted(self.best_results, key=lambda x: x[0], reverse=True)
            
            # 保存为JSON格式
            data = [
                {
                    'rank': i + 1,
                    'reward': reward,
                    'iteration': iteration,
                    'info': info
                }
                for i, (reward, iteration, info) in enumerate(sorted_results[:self.top_k])
            ]
            
            with open(self.best_results_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    def update(self, iteration, reward, info=None):
        """
        更新训练结果
        
        Args:
            iteration: 当前迭代次数
            reward: 当前奖励
            info: 额外信息（字典格式，如 {'protected_facilities': 5, 'intercepted_missiles': 80}）
        """
        if info is None:
            info = {}
        
        with self.lock:
            # 更新绘图数据
            self.plot_data['iterations'].append(iteration)
            self.plot_data['rewards'].append(reward)
            self.plot_data['protected_facilities'].append(info.get('protected_facilities', 0))
            self.plot_data['intercepted_missiles'].append(info.get('intercepted_missiles', 0))
            
            # 更新最好的K个结果
            if len(self.best_results) < self.top_k:
                heapq.heappush(self.best_results, (reward, iteration, info))
            elif reward > self.best_results[0][0]:  # 如果当前结果比最差的好
                heapq.heapreplace(self.best_results, (reward, iteration, info))
                self._save_best_results()
            else:
                self._save_best_results()  # 即使没有更新也保存一次
    
    def init_plot(self):
        """初始化实时绘图"""
        if self.fig is None:
            plt.ion()  # 开启交互模式
            self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
            self.fig.suptitle('训练过程实时监控', fontsize=16)
            
            # 设置子图标题
            self.axes[0, 0].set_title('奖励变化')
            self.axes[0, 0].set_xlabel('迭代次数')
            self.axes[0, 0].set_ylabel('奖励')
            self.axes[0, 0].grid(True)
            
            self.axes[0, 1].set_title('保护目标存活数')
            self.axes[0, 1].set_xlabel('迭代次数')
            self.axes[0, 1].set_ylabel('存活数量')
            self.axes[0, 1].grid(True)
            
            self.axes[1, 0].set_title('拦截导弹数量')
            self.axes[1, 0].set_xlabel('迭代次数')
            self.axes[1, 0].set_ylabel('拦截数量')
            self.axes[1, 0].grid(True)
            
            self.axes[1, 1].set_title('最好的10次结果')
            self.axes[1, 1].set_xlabel('排名')
            self.axes[1, 1].set_ylabel('奖励')
            self.axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.show(block=False)
    
    def update_plot(self):
        """更新实时绘图"""
        if self.fig is None:
            self.init_plot()
        
        with self.lock:
            if len(self.plot_data['iterations']) == 0:
                return
            
            iterations = list(self.plot_data['iterations'])
            rewards = list(self.plot_data['rewards'])
            protected = list(self.plot_data['protected_facilities'])
            intercepted = list(self.plot_data['intercepted_missiles'])
            
            # 更新奖励图
            self.axes[0, 0].clear()
            self.axes[0, 0].plot(iterations, rewards, 'b-', alpha=0.7)
            self.axes[0, 0].set_title('奖励变化')
            self.axes[0, 0].set_xlabel('迭代次数')
            self.axes[0, 0].set_ylabel('奖励')
            self.axes[0, 0].grid(True)
            
            # 更新保护目标图
            self.axes[0, 1].clear()
            self.axes[0, 1].plot(iterations, protected, 'g-', alpha=0.7)
            self.axes[0, 1].set_title('保护目标存活数')
            self.axes[0, 1].set_xlabel('迭代次数')
            self.axes[0, 1].set_ylabel('存活数量')
            self.axes[0, 1].set_ylim(-0.5, 8.5)
            self.axes[0, 1].grid(True)
            
            # 更新拦截数量图
            self.axes[1, 0].clear()
            self.axes[1, 0].plot(iterations, intercepted, 'r-', alpha=0.7)
            self.axes[1, 0].set_title('拦截导弹数量')
            self.axes[1, 0].set_xlabel('迭代次数')
            self.axes[1, 0].set_ylabel('拦截数量')
            self.axes[1, 0].grid(True)
            
            # 更新最好的10次结果
            self.axes[1, 1].clear()
            if len(self.best_results) > 0:
                sorted_results = sorted(self.best_results, key=lambda x: x[0], reverse=True)
                top_rewards = [r[0] for r in sorted_results[:self.top_k]]
                ranks = list(range(1, len(top_rewards) + 1))
                self.axes[1, 1].bar(ranks, top_rewards, color='orange', alpha=0.7)
                self.axes[1, 1].set_title('最好的10次结果')
                self.axes[1, 1].set_xlabel('排名')
                self.axes[1, 1].set_ylabel('奖励')
                self.axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)  # 短暂暂停以更新图形


class CheckpointManager:
    """检查点管理器：保存训练过程中最好的5个模型"""
    
    def __init__(self, checkpoint_dir="./checkpoints", top_k=5):
        self.checkpoint_dir = checkpoint_dir
        self.top_k = top_k
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 存储 (reward, checkpoint_path, iteration) 的最小堆
        self.best_checkpoints = []
        self.checkpoint_info_file = os.path.join(checkpoint_dir, "checkpoint_info.json")
        
        # 加载历史检查点信息
        self._load_checkpoint_info()
    
    def _load_checkpoint_info(self):
        """加载历史检查点信息"""
        if os.path.exists(self.checkpoint_info_file):
            try:
                with open(self.checkpoint_info_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.best_checkpoints = [
                        (item['reward'], item['checkpoint_path'], item['iteration'])
                        for item in data
                    ]
                    heapq.heapify(self.best_checkpoints)
            except Exception as e:
                print(f"加载检查点信息失败: {e}")
                self.best_checkpoints = []
    
    def _save_checkpoint_info(self):
        """保存检查点信息"""
        sorted_checkpoints = sorted(self.best_checkpoints, key=lambda x: x[0], reverse=True)
        data = [
            {
                'rank': i + 1,
                'reward': reward,
                'checkpoint_path': path,
                'iteration': iteration
            }
            for i, (reward, path, iteration) in enumerate(sorted_checkpoints)
        ]
        
        with open(self.checkpoint_info_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def should_save_checkpoint(self, reward):
        """判断是否应该保存检查点"""
        if len(self.best_checkpoints) < self.top_k:
            return True
        return reward > self.best_checkpoints[0][0]
    
    def save_checkpoint(self, checkpoint_path, reward, iteration):
        """
        保存检查点（如果进入最好的K个）
        
        Args:
            checkpoint_path: 检查点路径（ray会自动生成）
            reward: 当前奖励
            iteration: 当前迭代次数
        
        Returns:
            bool: 是否成功保存
        """
        if not self.should_save_checkpoint(reward):
            return False
        
        # 如果堆已满，删除最差的
        if len(self.best_checkpoints) >= self.top_k:
            worst_reward, worst_path, worst_iter = heapq.heappop(self.best_checkpoints)
            # 注意：这里不删除文件，因为ray可能还在使用
        
        # 添加新的检查点
        heapq.heappush(self.best_checkpoints, (reward, checkpoint_path, iteration))
        self._save_checkpoint_info()
        
        return True
