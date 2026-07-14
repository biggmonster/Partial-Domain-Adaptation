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


# D:\LJ\workstation\Vscode\New_ISRA\results\GASF\目标域随机四类\2_0_1_version.txt
# 读取日志文件

# root_dir = r'D:\LJ\workstation\Vscode\New_ISRA\results\GASF'

# subsub_dir = [ 'target_random_4class',
#             'target_fixed_4class',
#             ]
# root_dir = os.path.join(root_dir, subsub_dir[0])
# sub_dir = ['2_0']

# file_path_1_0 = os.path.join(root_dir, sub_dir[0],'2_0_1_version.txt')
# with open(file_path_1_0, 'r', encoding='utf-8') as f:
#     log_content_1_0 = f.read()

# file_path_2_0 = os.path.join(root_dir, sub_dir[0],'2_0_2_version.txt')
# with open(file_path_2_0, 'r', encoding='utf-8') as f:
#     log_content_2_0 = f.read()

# file_path_3_0 = os.path.join(root_dir, sub_dir[0],'2_0_3_version.txt')
# with open(file_path_3_0, 'r', encoding='utf-8') as f:
#     log_content_3_0 = f.read()

# file_path_4_0 = os.path.join(root_dir, sub_dir[0],'2_0_4_version.txt')
# with open(file_path_4_0, 'r', encoding='utf-8') as f:
#     log_content_4_0 = f.read()

# file_path_5_0 = os.path.join(root_dir, sub_dir[0],'2_0_5_version.txt')
# with open(file_path_5_0, 'r', encoding='utf-8') as f:
#     log_content_5_0 = f.read()

# file_path_6_0 = os.path.join(root_dir, sub_dir[0],'2_0_6_version.txt')
# with open(file_path_6_0, 'r', encoding='utf-8') as f:
#     log_content_6_0 = f.read()

# file_path_7_0 = os.path.join(root_dir, sub_dir[0],'2_0_7_version.txt')
# with open(file_path_7_0, 'r', encoding='utf-8') as f:
#     log_content_7_0 = f.read()

# file_path_8_0 = os.path.join(root_dir, sub_dir[0],'2_0_8_version.txt')
# with open(file_path_8_0, 'r', encoding='utf-8') as f:
#     log_content_8_0 = f.read()

# file_path_9_0 = os.path.join(root_dir, sub_dir[0],'2_0_9_version.txt')
# with open(file_path_9_0, 'r', encoding='utf-8') as f:
#     log_content_9_0 = f.read()


root_dir = r'D:\LJ\workstation\Vscode\New_ISRA\results\GASF'

subsub_dir = [ 'target_random_4class',
            'target_fixed_4class',
            ]
root_dir = os.path.join(root_dir, subsub_dir[0])
sub_dir = ['0_2']

file_path_1_0 = os.path.join(root_dir, sub_dir[0],'0_2_1_version.txt')
with open(file_path_1_0, 'r', encoding='utf-8') as f:
    log_content_1_0 = f.read()

file_path_2_0 = os.path.join(root_dir, sub_dir[0],'0_2_2_version.txt')
with open(file_path_2_0, 'r', encoding='utf-8') as f:
    log_content_2_0 = f.read()

file_path_3_0 = os.path.join(root_dir, sub_dir[0],'0_2_3_version.txt')
with open(file_path_3_0, 'r', encoding='utf-8') as f:
    log_content_3_0 = f.read()

file_path_4_0 = os.path.join(root_dir, sub_dir[0],'0_2_4_version.txt')
with open(file_path_4_0, 'r', encoding='utf-8') as f:
    log_content_4_0 = f.read()

file_path_5_0 = os.path.join(root_dir, sub_dir[0],'0_2_5_version.txt')
with open(file_path_5_0, 'r', encoding='utf-8') as f:
    log_content_5_0 = f.read()

file_path_6_0 = os.path.join(root_dir, sub_dir[0],'0_2_6_version.txt')
with open(file_path_6_0, 'r', encoding='utf-8') as f:
    log_content_6_0 = f.read()

file_path_7_0 = os.path.join(root_dir, sub_dir[0],'0_2_7_version.txt')
with open(file_path_7_0, 'r', encoding='utf-8') as f:
    log_content_7_0 = f.read()

file_path_8_0 = os.path.join(root_dir, sub_dir[0],'0_2_8_version.txt')
with open(file_path_8_0, 'r', encoding='utf-8') as f:
    log_content_8_0 = f.read()

file_path_9_0 = os.path.join(root_dir, sub_dir[0],'0_2_9_version.txt')
with open(file_path_9_0, 'r', encoding='utf-8') as f:
    log_content_9_0 = f.read()


# 提取数据1
data_pairs1 = extract_iter_precision(log_content_1_0) 
data_pairs2 = extract_iter_precision(log_content_2_0) 
data_pairs3 = extract_iter_precision(log_content_3_0) 
data_pairs4 = extract_iter_precision(log_content_4_0) 
data_pairs5 = extract_iter_precision(log_content_5_0) 
data_pairs6 = extract_iter_precision(log_content_6_0) 
data_pairs7 = extract_iter_precision(log_content_7_0) 
data_pairs8 = extract_iter_precision(log_content_8_0) 
data_pairs9 = extract_iter_precision(log_content_9_0) 
 
# # 打印结果 
# print("提取到的(iter, precision)数据对:") 
# print("-" * 30) 
# for iter_val, precision_val in data_pairs: 
#     print(f"迭代: {iter_val} -> 精度: {precision_val:.5f}")


# 提取并转换数据2
x = data_pairs1[0]
iters = [pair[0] for pair in data_pairs1]

precision1 = [pair[1] for pair in data_pairs1]
precision2 = [pair[1] for pair in data_pairs2]
precision3 = [pair[1] for pair in data_pairs3]
precision4 = [pair[1] for pair in data_pairs4]
precision5 = [pair[1] for pair in data_pairs5]
precision6 = [pair[1] for pair in data_pairs6]
precision7 = [pair[1] for pair in data_pairs7]
precision8 = [pair[1] for pair in data_pairs8]
precision9 = [pair[1] for pair in data_pairs9]

# 创建图表
plt.figure(figsize=(12, 8))  # 设置画布大小
# 设置美观的配色方案
colors = plt.cm.tab10(np.linspace(0, 1, 9))

# # 绘制7条折线，使用不同的颜色、标记和线型
# plt.plot(iters, precision1, marker='o', linestyle='-', color=colors[0], 
#          linewidth=2, markersize=6, alpha=0.8, label='SNR= 5')
# plt.plot(iters, precision2, marker='s', linestyle='-', color=colors[1], 
#          linewidth=2, markersize=6, alpha=0.8, label='SNR= 10')
# plt.plot(iters, precision3, marker='^', linestyle='-', color=colors[2], 
#          linewidth=2, markersize=6, alpha=0.8, label='SNR= 15')
# plt.plot(iters, precision4, marker='D', linestyle='-', color=colors[3], 
#          linewidth=2, markersize=6, alpha=0.8, label='SNR= 20')
# plt.plot(iters, precision5, marker='v', linestyle='-', color=colors[4], 
#          linewidth=2, markersize=6, alpha=0.8, label='SNR= 25')
# plt.plot(iters, precision6, marker='*', linestyle='-', color=colors[5], 
#          linewidth=2, markersize=8, alpha=0.8, label='SNR= -5')
# plt.plot(iters, precision7, marker='X', linestyle='-', color=colors[6], 
#          linewidth=2, markersize=7, alpha=0.8, label='SNR= -10')


# 绘制9条折线，使用不同的颜色、标记和线型
plt.plot(iters, precision1, marker='o', linestyle='-', color=colors[0], 
         linewidth=2, markersize=6, alpha=0.8, label='[0,1,2,3]')
plt.plot(iters, precision2, marker='s', linestyle='-', color=colors[1], 
         linewidth=2, markersize=6, alpha=0.8, label='[4,7,8,9]')
plt.plot(iters, precision3, marker='^', linestyle='-', color=colors[2], 
         linewidth=2, markersize=6, alpha=0.8, label='[0,1,2,5]')
plt.plot(iters, precision4, marker='D', linestyle='-', color=colors[3], 
         linewidth=2, markersize=6, alpha=0.8, label='[0,2,6,9]')
plt.plot(iters, precision5, marker='v', linestyle='-', color=colors[4], 
         linewidth=2, markersize=6, alpha=0.8, label='[3,4,8,9]')
plt.plot(iters, precision6, marker='*', linestyle='-', color=colors[5], 
         linewidth=2, markersize=8, alpha=0.8, label='[1,2,4,5]')
plt.plot(iters, precision7, marker='X', linestyle='-', color=colors[6], 
         linewidth=2, markersize=7, alpha=0.8, label='[0,2,4,8]')
plt.plot(iters, precision6, marker='P', linestyle='-', color=colors[7], 
         linewidth=2, markersize=8, alpha=0.8, label='[2,4,5,9]')
plt.plot(iters, precision7, marker='d', linestyle='-', color=colors[8], 
         linewidth=2, markersize=7, alpha=0.8, label='[0,4,5,9]')



# 设置标题和标签
plt.title("Source class=[0~9] SNR= initial, Target SNR = 10 ", fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("precision", fontsize=12)

# 添加图例
plt.legend(fontsize=12, loc='best', frameon=True, shadow=True, fancybox=True,
           title='Target Class', title_fontsize=13)

# 设置网格和刻度
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(0, 6000+1, 200), rotation=45)  # 每200单位一个刻度
plt.yticks([i*0.1 for i in range(0, 11)])  # 每0.1一个刻度

# 优化布局
plt.tight_layout() 

# 显示图表
plt.show()