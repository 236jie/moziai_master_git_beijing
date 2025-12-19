"""
训练结果后处理脚本：从Ray Tune结果中提取最好的10次训练结果
"""
import os
import json
import glob
from pathlib import Path


def extract_best_results(result_dir, output_file="./training_logs/best_results.json", top_k=10):
    """
    从Ray Tune结果目录中提取最好的K次训练结果
    
    Args:
        result_dir: Ray Tune结果目录
        output_file: 输出文件路径
        top_k: 提取最好的K次结果
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    best_results = []
    
    # 查找所有trial的结果文件
    if result_dir and os.path.exists(result_dir):
        for trial_dir in glob.glob(os.path.join(result_dir, "*")):
            if not os.path.isdir(trial_dir):
                continue
            
            # 查找result.json文件
            result_json = os.path.join(trial_dir, "result.json")
            if os.path.exists(result_json):
                try:
                    with open(result_json, 'r', encoding='utf-8') as f:
                        # result.json可能包含多行JSON
                        for line in f:
                            if line.strip():
                                try:
                                    result = json.loads(line)
                                    iteration = result.get('training_iteration', 0)
                                    reward = result.get('episode_reward_mean', 0)
                                    info = {
                                        'protected_facilities': result.get('custom_metrics', {}).get('protected_facilities_mean', 0),
                                        'intercepted_missiles': result.get('custom_metrics', {}).get('intercepted_missiles_mean', 0),
                                        'reward_max': result.get('episode_reward_max', 0),
                                        'reward_min': result.get('episode_reward_min', 0),
                                    }
                                    best_results.append((reward, iteration, info))
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    print(f"读取 {result_json} 时出错: {e}")
                    continue
    
    # 排序并取最好的K个
    best_results.sort(key=lambda x: x[0], reverse=True)
    top_k_results = best_results[:top_k]
    
    # 保存到文件
    data = [
        {
            'rank': i + 1,
            'reward': reward,
            'iteration': iteration,
            'info': info
        }
        for i, (reward, iteration, info) in enumerate(top_k_results)
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"已提取最好的{len(top_k_results)}次训练结果到 {output_file}")
    for i, (reward, iteration, info) in enumerate(top_k_results):
        print(f"  排名{i+1}: 轮数={iteration}, 奖励={reward:.2f}, "
              f"保护目标={info.get('protected_facilities', 0):.1f}, "
              f"拦截数量={info.get('intercepted_missiles', 0):.1f}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        result_dir = sys.argv[1]
    else:
        result_dir = None  # 需要手动指定
    extract_best_results(result_dir)
