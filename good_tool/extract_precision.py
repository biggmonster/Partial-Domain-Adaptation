import torch
import torchvision
import numpy as np
import os
import matplotlib.pyplot as plt
import re

def extract_iter_precision(log_content):
    """
    从日志内容中提取(iter, precision)数据对
    
    参数:
        log_content (str): 日志文本内容
        
    返回:
        list: 包含(iter, precision)元组的列表
    """
    # 匹配模式：查找包含"iter: XXXXX, precision: Y.YY"的行
    pattern = r"iter: (\d+), precision: (\d\.\d+)"
    matches = re.findall(pattern, log_content)
    
    # 将匹配结果转换为整数和浮点数，并将iter值格式化为5位数字, 保存为元组
    results = [(int(match[0]), float(match[1])) for match in matches]

    return results


root_dir_plug_EMA = r'D:\LJ\workstation\Vscode\New_ISRA\results\GASF\target_random_4class\plug_EMA\SNR_initial_SNR_10'
root_dir_no_EMA =   r'D:\LJ\workstation\Vscode\New_ISRA\results\GASF\target_random_4class\SNR_initial_to_SNR_10\12000_iterations'

subsub_dir = [ '0_2_1_version.txt','0_2_2_version.txt','0_2_3_version.txt',
               '0_2_4_version.txt','0_2_5_version.txt','0_2_6_version.txt',
               '0_2_7_version.txt','0_2_8_version.txt','0_2_9_version.txt'
            ]
all_data = []  # 存储每个文件的数据对列表

for i in range(9):
    file_path= os.path.join(root_dir_no_EMA, subsub_dir[i])
    with open(file_path, 'r', encoding='utf-8') as f:
        log_content_1_0 = f.read()

    data_pairs = extract_iter_precision(log_content_1_0)
    # print(data_pairs)
    print(f"File {i}: {subsub_dir[i]} has {len(data_pairs)} data points.")
    all_data.append(data_pairs)  

# 创建一个大图
plt.figure(figsize=(15,6))


marker=[
    'o', 's', '^', 'D', 'v', '*', 'X', 'P', 'd'
]
colors = plt.cm.tab10(np.linspace(0, 1, 9))

# 同时用枚举可以获取索引
for idx, data in enumerate(all_data):
    # 将数据对拆分成x和y两个列表
    x_vals = [pair[0] for pair in data]
    y_vals = [pair[1] for pair in data]
    # 绘制折线，用subsub_dir[idx]作为标签
    plt.plot(x_vals, y_vals, marker[idx], linestyle='-', color=colors[idx],
             linewidth=2, markersize=6, alpha=0.8, label=subsub_dir[idx]
             )

# 设置标题和标签
plt.title("no_EMA---Source class={0~9} SNR=initial, Target_SNR = 10", fontsize=16)
plt.xlabel("Iterations", fontsize=12)
plt.ylabel("precision", fontsize=12 )

# 添加图例
plt.legend(fontsize=12, loc='best', frameon=True, shadow=True, fancybox=True,
           title='Target Class', title_fontsize=13)

# 优化布局
plt.tight_layout() 

# 显示图表
plt.show()
