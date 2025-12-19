# 作者： 汪斌
# 文件： system_utils.py
# 说明： 系统工具
# 版权： 北京华戍防务技术有限公司

import inspect
import os
import re
import time
import tkinter as tk
from datetime import datetime

import psutil


def kill_moziserver():
    """关闭墨子进程，并删除Logs目录下的所有日志文件"""
    # 获取墨子进程并关闭
    if mozi_proc := next(
            (proc for proc in psutil.process_iter() if proc.name() == "MoziServer.exe"),
            None,
    ):
        mozi_proc.kill()
        time.sleep(1)
    # 删除Logs文件夹下所有文件
    mozi_path = os.environ["MOZIPATH"]
    log_path = os.path.join(mozi_path, "Logs")
    delete_files(log_path)


def delete_files(path):
    """递归式删除指定路径下的所有文件及文件夹"""
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            # 如果路径是文件或符号链接，则直接删除
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            # 如果路径是子文件夹，则递归式删除子文件夹中的所有文件及文件夹
            elif os.path.isdir(file_path):
                delete_files(file_path)  # 递归式删除子文件夹中的所有文件及文件夹
                os.rmdir(file_path)  # 删除空文件夹
        except:
            show_message("文件删除失败")


def show_message(message, font_name="微软雅黑", font_size=20, duration=2000):
    """
    在屏幕正中心打印（显示）所需信息
    默认使用微软雅黑字体、20号字号，显示2000毫秒（2秒）
    """
    root = tk.Tk()  # 创建Tkinter根窗口对象
    root.overrideredirect(True)  # 设置无边框
    root.attributes("-topmost", True)  # 设置置顶显示
    # root.withdraw()  # 暂时隐藏窗口

    # 创建一个Label控件，指定其父容器为root，并设置自定义文本及其字体、字号
    label = tk.Label(root, text=message, font=(font_name, font_size))
    # 将Label对象（label）放置在窗口对象（root）中，并自动管理其布局
    label.pack()

    # （根据Label对象）强制更新窗口布局，确保窗口（及控件）的尺寸、位置已完成调整，并可准确获取
    root.update_idletasks()

    # 获取屏幕（显示器）的宽度和高度，单位为像素
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    # 获取当前窗口的宽度和高度，单位为像素
    window_width = root.winfo_width()
    window_height = root.winfo_height()

    # 计算居中后的窗口左上角相对于屏幕左上角的水平位置和垂直位置，单位为像素
    x_pos = (screen_width - window_width) // 2  # 计算水平位置
    y_pos = (screen_height - window_height) // 2  # 计算垂直位置

    # 设置窗口左上角相对屏幕左上角的水平位置和垂直位置，以使窗口居中显示
    # 具体地，geometry方法的参数格式为"{宽度}x{高度}+{水平位置}+{垂直位置}"
    root.geometry(f"+{x_pos}+{y_pos}")
    # # 显示窗口，解除窗口隐藏状态
    # root.deiconify()

    # 在指定时间后自动执行root.destroy()，即在duration毫秒后销毁窗口
    root.after(duration, root.destroy)
    # 启动Tkinter的事件循环，使窗口保持显示状态直到（duration毫秒后）窗口被销毁（root.destroy()被执行）
    root.mainloop()

    # 返回原始字符串对象，用于后续调用
    # 如log_info(show_message("敌方雷达站被摧毁，我方获胜"))
    # 或assert alive_count <= self.alive_count[agent_id], log_info(show_message("编队内无人机数量增加，请检查代码逻辑"))
    # 和show_message函数一样，log_info函数的返回值亦为其参数对象，可作为assert语句的可选表达式
    return message


def pad_num(s, full_return=True):
    """
    用于正则表达式提取字符串后缀中的数字字符，并以将其补全至3位（左侧填充零）
    该函数用作字符串排序的键，以确保字符串排序时将字符串"2"排序到字符串"10"之前
    """
    padded_num = re.findall(r"\d+", s)[-1].zfill(3)
    if full_return:
        # 返回填充后的完整字符串
        return re.sub(r"\d+$", padded_num, s)
    # 返回数字字符串后缀
    return padded_num


def log_info(info=None):
    """
    自定义日志函数，用于在强化学习环境中记录报错信息及用户自定义信息，监测指定变量
    同时记录当前想定轮次及推演时间步，以及系统资源占用情况，以便debug以使训练持续进行
    注意，此函数涉及对环境类实例及想定轮次、推演时间步的获取，故仅能在环境类中使用
    用户可自行添加条件判断，将其扩展至其他使用场景
    """

    # 获取当前时间
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")

    # 获取调用该函数的方法的字符串名、环境类实例，以及当前想定轮次、推演时间步
    caller_frame = inspect.currentframe().f_back  # 获取调用当前函数的上一级函数（环境类实例的方法）的调用帧
    caller_name = caller_frame.f_code.co_name  # 获取调用此函数的外部方法名
    arg_value_map = inspect.getargvalues(caller_frame)[-1]  # 获取调用帧中所有局部变量与其对应对象的映射字典
    env = arg_value_map["self"]  # 获取强化学习环境类实例
    scen_steps, steps = env.scen_steps, env.steps  # 获取当前想定轮次及推演时间步

    # 获取当前系统资源占用情况
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent

    # 将相关信息写入自定义日志文件
    with open("info_log.txt", "a") as f:
        if isinstance(info, BaseException):  # 记录报错信息
            f.write(
                f"{current_time}: Warning! {type(info).__name__} occurred, "
                f"in {caller_name}, at {scen_steps} scen steps, {steps} steps.\n"
                f"Details: {str(info)}.\n"
            )
        elif isinstance(info, str):  # 记录用户自定义信息
            f.write(
                f"{current_time}: Recording user specified message, "
                f"in {caller_name}, at {scen_steps} scen steps, {steps} steps.\n"
                f"Message: {info}.\n"
            )
        elif info is None:  # 记录空条目
            f.write(
                f"{current_time}: Recording blank entry, "
                f"in {caller_name}, at {scen_steps} scen steps, {steps} steps.\n"
            )
        else:  # 记录变量值
            # 获取调用字符串，形如"log_info(usv_cnt)"，或"log_info(self.state)"，或"log_info(group_leader.dLatitude)"
            call_string = inspect.getframeinfo(caller_frame).code_context[0].strip()
            # 获取调用字符串小括号内的字符，形如"self.max_obs"或"usv_cnt"
            pattern = r"log_info\(([^)]+)\)"
            match = re.search(pattern, call_string)
            if match:
                var = "variable"  # 标识该变量是否为变量（"variable"）或属性（"attribute"）
                var_name = match.group(1).strip()  # 该变量的字符串名，即小括号内的字符串
                if "self" in var_name:
                    var = "attribute"  # 将该变量标识为属性
                    var_name = var_name.split(".")[-1]  # 提取该属性的字符串名
                f.write(
                    f"{current_time}: Recording user specified {var}, "
                    f"in {caller_name}, at {scen_steps} scen steps, {steps} steps.\n"
                    f"{var.capitalize()}: {var_name}, value: {info}.\n"  # info为该变量名所指向的具体对象
                )
            else:  # 极少情况下，match = None，执行var_name = match.group(1).strip()时会报错
                f.write(
                    f"{current_time}: Recording user specified variable, "
                    f"in {caller_name}, at {scen_steps} scen steps, {steps} steps.\n"
                    f"call_string: {call_string}, value: {info}.\n"  # info为该变量名所指向的具体对象
                )
        # 记录当前CPU及内存占用量
        if memory_usage > 90:
            f.write("Warning! Memory usage exceeds 90%.\n")
        f.write(
            f"Current CPU usage: {cpu_usage}%, current memory usage: {memory_usage}%.\n\n"
        )

    # 返回参数对象，用于后续调用
    return info
