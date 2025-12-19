"""
训练过程实时监控脚本
在训练过程中运行此脚本来实时查看训练效果

使用方法：
python monitor_training.py [result_dir]

或者：
python monitor_training.py  # 会自动查找最新的ray results目录
"""
import os
import json
import glob
import time
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path


def find_latest_ray_results():
    """查找最新的Ray Tune结果目录"""
    # 常见的Ray Tune结果目录
    possible_dirs = [
        os.path.expanduser("~/ray_results"),
        "./ray_results",
        "C:/Users/*/ray_results",
    ]
    
    latest_dir = None
    latest_time = 0
    
    for pattern in possible_dirs:
        for result_dir in glob.glob(pattern):
            if os.path.isdir(result_dir):
                # 查找最新的trial目录
                for trial_dir in glob.glob(os.path.join(result_dir, "*")):
                    if os.path.isdir(trial_dir):
                        mtime = os.path.getmtime(trial_dir)
                        if mtime > latest_time:
                            latest_time = mtime
                            latest_dir = trial_dir
    
    return latest_dir


def parse_result_json(result_file):
    """解析result.json文件"""
    results = []
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        result = json.loads(line)
                        results.append(result)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"读取 {result_file} 时出错: {e}")
    
    return results


def monitor_training(result_dir=None, update_interval=5):
    """实时监控训练过程"""
    if result_dir is None:
        result_dir = find_latest_ray_results()
    
    if result_dir is None:
        print("未找到Ray Tune结果目录，请手动指定：python monitor_training.py <result_dir>")
        return
    
    print(f"监控目录: {result_dir}")
    print("按Ctrl+C停止监控\n")
    
    # 初始化绘图
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('训练过程实时监控', fontsize=16)
    
    # 数据存储
    data = {
        'iterations': deque(maxlen=1000),
        'rewards': deque(maxlen=1000),
        'protected': deque(maxlen=1000),
        'intercepted': deque(maxlen=1000),
    }
    
    result_file = os.path.join(result_dir, "result.json")
    last_size = 0
    
    try:
        while True:
            # 检查文件是否有更新
            if os.path.exists(result_file):
                current_size = os.path.getsize(result_file)
                if current_size > last_size:
                    # 读取新数据
                    all_results = parse_result_json(result_file)
                    new_results = all_results[len(data['iterations']):]
                    
                    for result in new_results:
                        iteration = result.get('training_iteration', 0)
                        reward = result.get('episode_reward_mean', 0)
                        protected = result.get('custom_metrics', {}).get('protected_facilities_mean', 0)
                        intercepted = result.get('custom_metrics', {}).get('intercepted_missiles_mean', 0)
                        
                        data['iterations'].append(iteration)
                        data['rewards'].append(reward)
                        data['protected'].append(protected)
                        data['intercepted'].append(intercepted)
                    
                    last_size = current_size
                    
                    # 更新绘图
                    if len(data['iterations']) > 0:
                        iterations = list(data['iterations'])
                        rewards = list(data['rewards'])
                        protected = list(data['protected'])
                        intercepted = list(data['intercepted'])
                        
                        # 奖励图
                        axes[0, 0].clear()
                        axes[0, 0].plot(iterations, rewards, 'b-', alpha=0.7)
                        axes[0, 0].set_title('奖励变化')
                        axes[0, 0].set_xlabel('迭代次数')
                        axes[0, 0].set_ylabel('奖励')
                        axes[0, 0].grid(True)
                        
                        # 保护目标图
                        axes[0, 1].clear()
                        axes[0, 1].plot(iterations, protected, 'g-', alpha=0.7)
                        axes[0, 1].set_title('保护目标存活数')
                        axes[0, 1].set_xlabel('迭代次数')
                        axes[0, 1].set_ylabel('存活数量')
                        axes[0, 1].set_ylim(-0.5, 8.5)
                        axes[0, 1].grid(True)
                        
                        # 拦截数量图
                        axes[1, 0].clear()
                        axes[1, 0].plot(iterations, intercepted, 'r-', alpha=0.7)
                        axes[1, 0].set_title('拦截导弹数量')
                        axes[1, 0].set_xlabel('迭代次数')
                        axes[1, 0].set_ylabel('拦截数量')
                        axes[1, 0].grid(True)
                        
                        # 最近10次结果
                        axes[1, 1].clear()
                        if len(rewards) > 0:
                            recent_rewards = rewards[-10:]
                            recent_iterations = iterations[-10:]
                            axes[1, 1].bar(range(len(recent_rewards)), recent_rewards, color='orange', alpha=0.7)
                            axes[1, 1].set_title('最近10次训练结果')
                            axes[1, 1].set_xlabel('相对位置')
                            axes[1, 1].set_ylabel('奖励')
                            axes[1, 1].grid(True)
                        
                        plt.tight_layout()
                        plt.draw()
                        plt.pause(0.01)
                        
                        # 输出最新结果
                        if len(rewards) > 0:
                            print(f"迭代 {iterations[-1]}: 奖励={rewards[-1]:.2f}, "
                                  f"保护目标={protected[-1]:.1f}, 拦截数量={intercepted[-1]:.1f}")
            
            time.sleep(update_interval)
    
    except KeyboardInterrupt:
        print("\n监控已停止")
        plt.close('all')


if __name__ == '__main__':
    import sys
    result_dir = sys.argv[1] if len(sys.argv) > 1 else None
    monitor_training(result_dir)
