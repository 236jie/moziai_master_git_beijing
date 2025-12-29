#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新版训练结果统计脚本

功能：
- 读取终端输出日志（例如"训练结果终端输出.txt"）
- 按"局"（episode）进行划分
- 统计每一局：
    * 局号
    * 剩余保护目标数量
    * C-400 / HQ-9A / HQ-12 发射导弹数量（基于关键字符串统计）
    * 拦截成功数量（可选）
- 统计整体平均值，便于和"见弹就打"的基线对比

更新说明：
- 适配新的输出格式：-------发射了X枚C-400/HQ-9A/HQ-12导弹------
- 数字格式适配阿拉伯数字

使用方法（在 sdfk_test 目录下）：
    python 统计训练结果.py                 # 使用默认日志文件名：训练结果终端输出.txt
    python 统计训练结果.py other_log.txt   # 指定日志文件
"""
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class EpisodeStat:
    episode_idx: int
    protected_left: int
    c400_fired: int
    hq9_fired: int
    hq12_fired: int
    intercept_count: Optional[int] = None  # 拦截成功数量


def parse_episode_stats(lines: List[str]) -> List[EpisodeStat]:
    """
    从日志行列表中解析每一局的统计信息。
    约定：
      - 每局结束时有一行类似：'++++Score: ... step: ...'
      - 保护目标剩余：'红方地面核心设施还剩下X个'
      - 发射统计基于关键打印：
          * '执行C-400，打出X枚' => C-400 +X
          * '-------发射了X枚C-400导弹------' => C-400 +X
          * '执行HQ-9A，打出X枚' => HQ-9A +X
          * '-------发射了X枚HQ-9A导弹------' => HQ-9A +X
          * '执行HQ-12，打出X枚' => HQ-12 +X
          * '-------发射了X枚HQ-12导弹------' => HQ-12 +X
      - 拦截成功数量：'本局拦截成功数量: X'
    """
    episodes: List[EpisodeStat] = []

    current_idx = 1
    c400 = hq9 = hq12 = 0
    protected_left = None
    intercept_count = None

    # 正则预编译
    re_protected = re.compile(r"红方地面核心设施还剩下(\d+)个")
    re_c400_exec = re.compile(r"执行C-400，打出(\d+)枚")
    re_hq9_exec = re.compile(r"执行HQ-9A，打出(\d+)枚")
    re_hq12_exec = re.compile(r"执行HQ-12，打出(\d+)枚")

    # 新的发射格式正则
    re_c400_fire = re.compile(r"-+发射了(\d+)枚C-400导弹-+")
    re_hq9_fire = re.compile(r"-+发射了(\d+)枚HQ-9A导弹-+")
    re_hq12_fire = re.compile(r"-+发射了(\d+)枚HQ-12导弹-+")

    # 拦截数量正则
    re_intercept = re.compile(r"本局拦截成功数量:\s*(\d+)")

    for line in lines:
        text = line.strip()

        # 统计保护目标剩余
        m = re_protected.search(text)
        if m:
            try:
                protected_left = int(m.group(1))
            except ValueError:
                pass

        # 统计拦截成功数量
        m = re_intercept.search(text)
        if m:
            try:
                intercept_count = int(m.group(1))
            except ValueError:
                pass

        # 统计发射数量 - 旧格式（执行语句）
        m = re_c400_exec.search(text)
        if m:
            try:
                c400 += int(m.group(1))
            except ValueError:
                pass

        m = re_hq9_exec.search(text)
        if m:
            try:
                hq9 += int(m.group(1))
            except ValueError:
                pass

        m = re_hq12_exec.search(text)
        if m:
            try:
                hq12 += int(m.group(1))
            except ValueError:
                pass

        # 统计发射数量 - 新格式（发射语句）
        m = re_c400_fire.search(text)
        if m:
            try:
                c400 += int(m.group(1))
            except ValueError:
                pass

        m = re_hq9_fire.search(text)
        if m:
            try:
                hq9 += int(m.group(1))
            except ValueError:
                pass

        m = re_hq12_fire.search(text)
        if m:
            try:
                hq12 += int(m.group(1))
            except ValueError:
                pass

        # 局结束标记
        if "++++Score:" in text:
            if protected_left is None:
                # 如果没找到，默认 0，防止统计缺失
                protected_left_val = 0
            else:
                protected_left_val = protected_left

            ep = EpisodeStat(
                episode_idx=current_idx,
                protected_left=protected_left_val,
                c400_fired=c400,
                hq9_fired=hq9,
                hq12_fired=hq12,
                intercept_count=intercept_count,
            )
            episodes.append(ep)

            # 为下一局重置
            current_idx += 1
            c400 = hq9 = hq12 = 0
            protected_left = None
            intercept_count = None

    return episodes


def main():
    # 默认日志文件名
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = "训练结果终端输出.txt"

    if not os.path.isabs(log_path):
        # 相对路径：相对于当前脚本所在目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_path = os.path.join(base_dir, log_path)

    if not os.path.exists(log_path):
        print(f"找不到日志文件：{log_path}")
        return

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    episodes = parse_episode_stats(lines)
    if not episodes:
        print("没有解析到任何完整的局（未发现 '++++Score:' 行）。")
        return

    print(f"共解析到 {len(episodes)} 局训练结果：\n")
    print("每局统计（局号, 剩余保护目标, 拦截数量, C-400发射, HQ-9A发射, HQ-12发射）：")
    for ep in episodes:
        intercept_str = f"拦截: {ep.intercept_count}" if ep.intercept_count is not None else "拦截: N/A"
        print(f"第{ep.episode_idx:3d}局 | 保护目标剩余: {ep.protected_left} | {intercept_str:10} | "
              f"C-400: {ep.c400_fired:3d} | HQ-9A: {ep.hq9_fired:3d} | HQ-12: {ep.hq12_fired:3d}")

    # 整体平均值
    avg_protected = sum(e.protected_left for e in episodes) / len(episodes)
    avg_c400 = sum(e.c400_fired for e in episodes) / len(episodes)
    avg_hq9 = sum(e.hq9_fired for e in episodes) / len(episodes)
    avg_hq12 = sum(e.hq12_fired for e in episodes) / len(episodes)

    # 如果有拦截数据，计算平均值
    episodes_with_intercept = [e for e in episodes if e.intercept_count is not None]
    if episodes_with_intercept:
        avg_intercept = sum(e.intercept_count for e in episodes_with_intercept) / len(episodes_with_intercept)
    else:
        avg_intercept = None

    print("\n" + "=" * 60)
    print("整体平均统计（可与'见弹就打'基线对比）")
    print("=" * 60)
    print(f"平均剩余保护目标数: {avg_protected:.2f} / 8")
    if avg_intercept is not None:
        print(f"平均拦截成功数量: {avg_intercept:.2f}")
    print(f"平均 C-400 发射数量: {avg_c400:.2f}")
    print(f"平均 HQ-9A 发射数量: {avg_hq9:.2f}")
    print(f"平均 HQ-12 发射数量: {avg_hq12:.2f}")
    print(f"平均总发射数量: {avg_c400 + avg_hq9 + avg_hq12:.2f}")

    # 计算保护目标损失率
    initial_protected = 8  # 初始有8个保护目标
    loss_rate = (initial_protected - avg_protected) / initial_protected * 100
    print(f"平均保护目标损失率: {loss_rate:.2f}%")

    # 计算平均拦截效率（如果有拦截数据）
    if avg_intercept is not None and (avg_c400 + avg_hq9 + avg_hq12) > 0:
        efficiency = avg_intercept / (avg_c400 + avg_hq9 + avg_hq12) * 100
        print(f"平均拦截效率（拦截数/发射数）: {efficiency:.2f}%")

    # 同时导出到简易 CSV，方便后续画图或做进一步分析
    out_csv = os.path.join(os.path.dirname(log_path), "训练结果统计.csv")
    try:
        with open(out_csv, "w", encoding="utf-8") as f:
            # 写入标题行
            if episodes[0].intercept_count is not None:
                f.write("episode,protected_left,intercept_count,c400_fired,hq9_fired,hq12_fired\n")
            else:
                f.write("episode,protected_left,c400_fired,hq9_fired,hq12_fired\n")

            # 写入数据行
            for ep in episodes:
                if ep.intercept_count is not None:
                    f.write(f"{ep.episode_idx},{ep.protected_left},{ep.intercept_count},"
                            f"{ep.c400_fired},{ep.hq9_fired},{ep.hq12_fired}\n")
                else:
                    f.write(f"{ep.episode_idx},{ep.protected_left},"
                            f"{ep.c400_fired},{ep.hq9_fired},{ep.hq12_fired}\n")
        print(f"\n详细逐局数据已导出到：{out_csv}")
    except Exception as e:
        print(f"\n导出 CSV 时出错（不影响终端统计）：{e}")


if __name__ == "__main__":
    main()