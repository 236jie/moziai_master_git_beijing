"""
检查点管理脚本：从训练结果中提取最好的5个检查点

使用方法：
python manage_checkpoints.py [result_dir]
"""
import os
import json
import glob
import shutil
from pathlib import Path


def manage_best_checkpoints(result_dir, output_dir="./checkpoints/best_5", top_k=5):
    """
    从Ray Tune结果中提取最好的K个检查点
    
    Args:
        result_dir: Ray Tune结果目录
        output_dir: 输出目录
        top_k: 提取最好的K个检查点
    """
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoints_info = []
    
    if result_dir and os.path.exists(result_dir):
        # 查找所有trial目录
        for trial_dir in glob.glob(os.path.join(result_dir, "*")):
            if not os.path.isdir(trial_dir):
                continue
            
            # 查找checkpoint目录
            checkpoint_dirs = glob.glob(os.path.join(trial_dir, "checkpoint_*"))
            
            # 查找result.json来获取对应的奖励
            result_json = os.path.join(trial_dir, "result.json")
            if os.path.exists(result_json):
                try:
                    with open(result_json, 'r', encoding='utf-8') as f:
                        results_by_iteration = {}
                        for line in f:
                            if line.strip():
                                try:
                                    result = json.loads(line)
                                    iteration = result.get('training_iteration', 0)
                                    reward = result.get('episode_reward_mean', 0)
                                    results_by_iteration[iteration] = reward
                                except json.JSONDecodeError:
                                    continue
                    
                    # 匹配checkpoint和奖励
                    for checkpoint_dir in checkpoint_dirs:
                        checkpoint_name = os.path.basename(checkpoint_dir)
                        # 从checkpoint_123中提取123
                        try:
                            iteration = int(checkpoint_name.replace('checkpoint_', ''))
                            reward = results_by_iteration.get(iteration, 0)
                            checkpoints_info.append((reward, iteration, checkpoint_dir))
                        except ValueError:
                            continue
                except Exception as e:
                    print(f"处理 {trial_dir} 时出错: {e}")
    
    # 排序并选择最好的K个
    checkpoints_info.sort(key=lambda x: x[0], reverse=True)
    top_k_checkpoints = checkpoints_info[:top_k]
    
    # 复制检查点到输出目录
    checkpoint_list = []
    for i, (reward, iteration, checkpoint_dir) in enumerate(top_k_checkpoints):
        dest_dir = os.path.join(output_dir, f"rank_{i+1}_iter_{iteration}_reward_{reward:.2f}")
        try:
            if os.path.exists(checkpoint_dir):
                if os.path.exists(dest_dir):
                    shutil.rmtree(dest_dir)
                shutil.copytree(checkpoint_dir, dest_dir)
                checkpoint_list.append({
                    'rank': i + 1,
                    'reward': reward,
                    'iteration': iteration,
                    'checkpoint_path': dest_dir,
                })
                print(f"✓ 复制检查点 {i+1}: 奖励={reward:.2f}, 迭代={iteration}, 路径={dest_dir}")
        except Exception as e:
            print(f"复制检查点 {checkpoint_dir} 时出错: {e}")
    
    # 保存检查点信息
    info_file = os.path.join(output_dir, "checkpoint_info.json")
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_list, f, indent=2, ensure_ascii=False)
    
    print(f"\n保存最好的{len(checkpoint_list)}个检查点到 {output_dir}")
    print(f"检查点信息保存到 {info_file}")


if __name__ == '__main__':
    import sys
    result_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if result_dir is None:
        # 尝试查找最新的ray results
        possible_dirs = [
            os.path.expanduser("~/ray_results"),
            "./ray_results",
        ]
        for base_dir in possible_dirs:
            if os.path.exists(base_dir):
                # 查找最新的trial
                trials = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
                if trials:
                    result_dir = max(trials, key=os.path.getmtime)
                    break
    
    if result_dir:
        manage_best_checkpoints(result_dir)
    else:
        print("请指定结果目录：python manage_checkpoints.py <result_dir>")
