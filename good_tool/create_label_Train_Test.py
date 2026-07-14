import os
import random

def generate_source_label_files(root_dir, styles, num_classes, output_dir, source_samples_num):
    """
    为ADS-B数据集的每个风格生成标签文件
    
    Args:
        root_dir (str): 数据集根目录，如 "ads-b"
        styles (list): 风格子文件夹列表
        num_classes (int): 每个风格下的类别数量--15
        output_dir (str): 输出标签文件的目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历每个风格(干扰模式)
    for style in styles:
        style_dir = os.path.join(root_dir, style)  # 通过.join()拼接，生成每个干扰模式各自的路径
        if not os.path.exists(style_dir):
            print(f"警告: 风格目录 '{style_dir}' 不存在，跳过")
            continue
            
        output_txt = os.path.join(output_dir, f"{style}_list.txt")   # 生成标签文件的绝对路径
        samples = []
        
        # 遍历每个类别
        for class_id in range(num_classes):
            if class_id < 9: 
                class_dir = os.path.join(style_dir, f"class_0{class_id+1}")
            else:
                class_dir = os.path.join(style_dir, f"class_{class_id+1}")
            if not os.path.exists(class_dir):
                print(f"警告: 类别目录 '{class_dir}' 不存在，跳过")
                continue
            t = 0
            # 获取类别目录下的所有文件
            for filename in os.listdir(class_dir):

                #------------------------------------这部分是为了360样本/50类，而单独加的
                t +=1       
                if t > source_samples_num:
                    break
                #-------------------------------------

                if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    # 构建绝对路径 
                    rel_path = os.path.join(class_dir,  filename)
                    # 确保使用正斜杠
                    rel_path = rel_path.replace('/', '\\')
                    # 添加到样本列表
                    samples.append(f"{rel_path} {class_id}\n")   # class_id是标签，标签从0开始，和文件夹名称无关
                    
        
        # 将样本标签写入标签文件
        with open(output_txt, 'w') as f:
            f.writelines(samples)
            
        print(f"已生成 {style} 风格的标签文件: {output_txt}，共 {len(samples)} 个样本")


def generate_target_label_files(root_dir, styles, output_dir_train, output_dir_test, target_samples_num, target_class_num, rate):
    """
    为ADS-B数据集的每个风格生成标签文件
    
    Args:
        root_dir (str): 数据集根目录，如 "ads-b"
        styles (list): 风格子文件夹列表
        num_classes (int): 每个风格下的类别数量--
        output_dir (str): 输出标签文件的目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir_train, exist_ok=True)
    os.makedirs(output_dir_test, exist_ok=True)
    
    # 遍历每个风格
    for style in styles:
        style_dir = os.path.join(root_dir, style)  # 通过.join()拼接，生成每个干扰模式各自的路径
        if not os.path.exists(style_dir):
            print(f"警告: 风格目录 '{style_dir}' 不存在111，跳过")
            continue
            
        output_txt_train = os.path.join(output_dir_train, f"{style}_{target_class_num}_list.txt")   # 生成标签文件的绝对路径
        output_txt_test = os.path.join(output_dir_test, f"{style}_{target_class_num}_list.txt")
        samples_train = []
        samples_test = []

        i = 0

        #---------------------------------------------------------------------------------------------------------------------------------------
        # 随机选择 30 个不重复的整数
        # random_class_ids = random.sample(range(50), 30)
        # print(random_class_ids)
        # [0,1,2,3]  [3,4,8,9]  [1,2,4,5]
        # random_class_ids =     [19, 27, 47, 18, 33, 48, 40, 13, 5, 23, 38, 31, 29, 43, 9, 36, 32, 15, 45, 2, 22, 21, 16, 46, 8, 7, 6, 35, 1, 3]
        ten_class_ids =        [19, 27, 47, 18, 33, 48, 40, 13, 5, 23]
        fifteen_class_ids =    [19, 27, 47, 18, 33, 48, 40, 13, 5, 23, 38, 31, 29, 43, 9]
        twenty_class_ids =     [19, 27, 47, 18, 33, 48, 40, 13, 5, 23, 38, 31, 29, 43, 9, 36, 32, 15, 45, 2]
        twenty_five_class_ids= [19, 27, 47, 18, 33, 48, 40, 13, 5, 23, 38, 31, 29, 43, 9, 36, 32, 15, 45, 2, 22, 21, 16, 46, 8]
        thirty_class_ids =     [19, 27, 47, 18, 33, 48, 40, 13, 5, 23, 38, 31, 29, 43, 9, 36, 32, 15, 45, 2, 22, 21, 16, 46, 8, 7, 6, 35, 1, 3]
        # [0,2,3,7]
        #---------------------------------------------------------------------------------------------------------------------------------------

        # random_class_ids = [0,2,3,7]

        for class_id in fifteen_class_ids:
            if class_id  < 9: 
                class_dir = os.path.join(style_dir, f"class_0{class_id+1}")
            else:
                class_dir = os.path.join(style_dir, f"class_{class_id+1}")

            
            i += 1  
            if i > target_class_num:   # 目标域类别数量 
                break

            if not os.path.exists(class_dir):
                print(f"警告: 类别目录 '{class_dir}' 不存在222, 跳过")
                continue
            
            class_samples_train = []  # 某一类别目录下的所有图像标签---临时保存
            class_samples_test = []
            class_samples = []
            
            # 获取类别目录下的所有文件
            for filename in os.listdir(class_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    # 构建绝对路径 
                    rel_path = os.path.join(class_dir,  filename)
                    # 
                    rel_path = rel_path.replace('\\', '/')
                    # 添加到样本列表
                    class_samples.append(f"{rel_path} {class_id}\n")   # class_id是标签，标签从0开始，和文件夹名称无关
                    
            # 如果样本数大于需要的数量，选择前指定数量的样本
            if len(class_samples) > target_samples_num:
                class_samples = class_samples[:target_samples_num]

            split_point = int(len(class_samples) * rate)         
            class_samples_train = class_samples[:split_point]
            class_samples_test = class_samples[split_point:]

            samples_train.extend(class_samples_train)
            samples_test.extend(class_samples_test)
            
        
        # 将样本标签写入标签文件
        with open(output_txt_train, 'w') as f:
            f.writelines(samples_train)
        print(f"已生成 {style} 风格的标签文件: {output_txt_train}，共 {len(samples_train)} 个样本")  

        with open(output_txt_test, 'w') as f:
            f.writelines(samples_test)        
        print(f"已生成 {style} 风格的标签文件: {output_txt_test}，共 {len(samples_test)} 个样本")



if __name__=='__main__':

#----------------------------LongSig_50------------------------------------------------------------------- 
    # ; source_root_dir   = r'data0/liangjie/ISRA/data/image_LongSig_50/source/gasf_old'           # 数据集图片文件夹路径
    # ; source_output_dir = r"D:\LJ\workstation\Vscode\New_ISRA\data\LongSig_50\Source\GASF_old\fixed_30class"         # 标签文件存放的路径

    target_root_dir   = r"/data0/liangjie/ISRA/data/image_LongSig_50/target/gasf_old"           # 数据集图片文件夹路径
    target_output_dir_train = r"/data0/liangjie/AR/data/label_LongSig_50/Target/Train/Train1_Test9"         # 标签文件存放的路径
    target_output_dir_test = r"/data0/liangjie/AR/data/label_LongSig_50/Target/Test/Train1_Test9"    
    
#----------------------------LongSig_10------------------------------------------------------------------- 
    # root_dir   = r"D:\LJ\workstation\Matlab\ads-b_data_gen\data_LongSig_10\gasf"              # 数据集图片文件夹路径
    # output_dir = r"D:\LJ\workstation\Vscode\New_ISRA\data\LongSig_10\GASF\fixed_4class"       # 标签文件存放的路径
    
# ---------------------------preamble----------------------------------------------------------------------
    # root_dir = r"D:\LJ\workstation\Matlab\ads-b_data_gen\data_LongSig_10\gasf\preamble\PAA_size_32"
    # output_dir = r"D:\LJ\workstation\Vscode\New_ISRA\data\LongSig_10\GASF\preamble\PAA_size_32"   # 标签文件存放的路径   


    # styles = ["X_15_100_0dB",                         # 所有的干扰模式
    #           "X_15_100_5dB", "X_15_100_-5dB",
    #           "X_15_100_10dB","X_15_100_-10dB",
    #           "X_15_100_15dB","X_15_100_-15dB",
    #           "X_15_100_20dB","X_15_100_-20dB",
    #           "X_15_100_25dB"
    #           ]    
    styles = [  
                 'SNR_0' 
                ] 
    # styles = [  'SNR_5_RayL' , 'SNR_10_RayL',
    #             'SNR_15_RayL', 'SNR_20_RayL',
    #             'SNR_0_RayL',  'SNR_-5_RayL', 
    #         ] 

#-----------源域10类，目标域4类---------------------------------------------------
    # num_classes = 10  
    # target_samples_num = 250 
    # target_class_num = 4
#-----------源域50类，目标域30类-----------------------------------------------------------------     
    num_classes = 50                       # 每个干扰模式下的类别数量
    source_samples_num = 360
    target_samples_num = 360               # -
    target_class_num = 15
    rate  = 0.1                           # Train 和 Test 的样本比例
#----------------------------------------------------------------------------
   
    # 生成标签文件
    # generate_source_label_files(source_root_dir, styles, num_classes, source_output_dir, source_samples_num)

    generate_target_label_files(target_root_dir, styles, target_output_dir_train, target_output_dir_test, target_samples_num, target_class_num, rate)

