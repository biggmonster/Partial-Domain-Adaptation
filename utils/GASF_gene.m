function [rgb_image] = GASF_gene(X, traget_size)
# X 单个ads-b信号序列，[序列长度，通道数]
#
# GASF是一种将一维时序数据映射到二维极坐标空间的方法，通过三角函数保留时序的幅值和相位信息。
# 1.平均平滑采样（分段聚合近似）
# 2.归一化：将时序数据归一化到 [-1, 1] 区间
# 3.极坐标映射：将归一化后的数据转换为角度
# 4.构建格拉姆矩阵--GASF

resize_length = traget_size;     # 目标图像尺寸N*N 
#step = sstep;    # 步长
cmap = turbo(256); # 生成了一个 ​256×3​ 的矩阵 turbo/parula

# 获取数据维度  
samples_IQ = X;

samples_I = samples_IQ(1,:, 1);  # 样本的第1通道，实数部分，长度为[序列长度]
samples_Q = samples_IQ(1,:, 2);  # 样本的第2通道，虚数部分，长度为[序列长度]


# 平滑采样，等价于分段聚合近似
samples_smooth_I = smoothDownsample(samples_I,  resize_length);
samples_smooth_Q = smoothDownsample(samples_Q,  resize_length);

#归一化[-1, 1]
samples_normalized_I = normalize_data(samples_smooth_I);
samples_normalized_Q = normalize_data(samples_smooth_Q);

#生成格拉姆矩阵
GASF_I = zeros(resize_length); # zeros(n) 生成一个 n×n 的零矩阵
GASF_Q = zeros(resize_length);
GASF_I(:,:) = generate_GASF(samples_normalized_I);
GASF_Q(:,:) = generate_GASF(samples_normalized_Q);

#合并双通道（水平拼接）
GASF_combined = zeros(resize_length, resize_length*2);
GASF_combined(:,:) = [GASF_I(:,:), GASF_Q(:,:)];

# 将归一化后的矩阵值（灰度索引）转换为 0 到 size(cmap, 1) 范围内的整数索引
index_matrix = round(GASF_combined * (size(cmap, 1) - 1)) + 1;
# 将归一化后的矩阵值（灰度索引）转换为 RGB 图像
rgb_image = ind2rgb(index_matrix, cmap);


# #双线性插值，1*512*256-->1*256*256
# resize_GASF_combined = imresize(GASF_combined, [256,256], 'bicubic');
# index_matrix = round(resize_GASF_combined * (size(cmap, 1) - 1)) + 1;
# rgb_image = ind2rgb(index_matrix, cmap);

end

##
# 用于减小时间序列长度（分段聚合近似PAA）
# -----------------------------平滑采样------------------在后续可以考虑更新---------------------
function sampleSignal = smoothDownsample(rawSignal,  Targetlength)

    # rawSignal：原始信号序列
    # Targetlength：分段平滑后的信号序列长度
    # step：窗口大小（floor向下取整）,不是窗口滑动的那种
    
    n = length(rawSignal);
    step = floor(n / Targetlength);     
    sampleSignal = zeros(Targetlength, 1);   # 创建空序列作为容器
    
    for i = 1:Targetlength
        if i < Targetlength
            end_idx = i * step;
        else
            end_idx = n;   # 最后一段取到末尾，为了防止原始信号的尾部不被忽略，强制调整最后一段的结束位置为信号末端n
        end
        start_idx = (i - 1) * step + 1;
        sampleSignal(i) = mean(rawSignal(start_idx:end_idx)); # 简单平均平滑采样  
    end
end

##
#-----------------------------归一化[-1, 1]---------------------------------------
function s_normalized = normalize_data(signal)
    s_min = min(signal);
    s_max = max(signal);
    if s_max == s_min
        s_normalized = zeros(size(signal));
    else
        # 标准归一化公式 [-1, 1]
        s_normalized = 2 * (signal - s_min) / (s_max - s_min) - 1;
    end
end

##
#---------------------------构建格拉姆角场---------------------------------
function gasf = generate_GASF(s_normalized)
    # 生成生成GASF，
    # 确保数值稳定性，范围[-1,1]
    s_normalized = min(max(s_normalized, -1), 1); 

    #--------------- 核心：保留符号的极性映射--------------------------
        # # 分离幅值和符号
        # magnitude = abs(s_normalized); # 余弦值的绝对值，
        # sign_val = sign(s_normalized); # 符号，具有正负，代表*****信息
        # # .* 表示逐元素相乘：theta范围[-pai/2, pai/2]
        # theta = acos(magnitude) .* sign_val; # 

        # 原始gasf计算公式：theta范围[0，pai]
        theta = acos(s_normalized);

    # 向量化计算格拉姆矩阵
    cos_sum = cos(theta + theta'); 
    gasf = (cos_sum + 1)/2; # 映射到[0,1]便于可视化
end