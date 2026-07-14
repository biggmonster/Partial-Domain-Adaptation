import argparse
import os, random, pdb, math, sys
from contextlib import redirect_stdout
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import my_loss, en_network
# import test_4
import lr_schedule, data_list
import torch.nn.functional as F
#from sklearn.metrics import confusion_matrix
import pandas as pd
from tqdm import tqdm
import torch.utils.checkpoint as checkpoint
from torch.utils.data import Subset
import densenet169


# 原代码，可用于ads-b信号      
def image_train(resize_size=256, crop_size=224):   # resize_size指定图像在裁剪前的尺寸，crop_size随机裁剪后图片的尺寸
    return  transforms.Compose([                   
        transforms.Resize((resize_size, resize_size)),    # 调整输入图像的尺寸为256  
        transforms.RandomCrop(crop_size),                 # 随机裁剪图像（默认224×224）——这是一种数据增强技术
        transforms.RandomHorizontalFlip(),                # 以50%的概率随机水平翻转图像——另一种数据增强技术        
        transforms.ToTensor(),                            # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # w这些特定的均值和标准差值是 ​ImageNet 数据集​ 上预训练模型的标准化参数，
                                                                                       # 用于将像素值转换到适合模型输入的范围内，与 PyTorch 的预训练模型兼容
                                                                                       # 迁移学习，应保持这些参数不变以保证模型正常工作。
    ])

def image_test(resize_size=256, crop_size=224):
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),          # 输入图像从中心开始裁剪出正方形区域  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def image_test_classification(loader, model):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, indices in loader["test"]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            _, logits = model(inputs)

            softplus_out = F.softplus(logits)
            softmax_out = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            for idx, label, pred, logit, sp, sm in zip(indices, labels, preds, logits, softplus_out, softmax_out):
                print(f"sample={idx.item()} label={label.item()} pred={pred.item()}")
                print(f"logits={format_values(logit)}")
                print(f"softplus={format_values(sp)}")
                print(f"softmax={format_values(sm)}")

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).long()

    predict = all_logits.argmax(dim=1)
    accuracy = (predict == all_labels).float().mean().item()
    mean_ent = my_loss.Entropy(F.softmax(all_logits, dim=1)).mean().item()

    print(f"test accuracy: {accuracy:.4f}")

    return accuracy, mean_ent

def format_values(values):
    return "[" + ", ".join(f"{value:.3f}" for value in values.cpu().tolist()) + "]"

def print_line():
    print('-'*100)

def print_args(args):
    log_str = ("==========================================\n")
    log_str += ("==========       config      =============\n")
    log_str += ("==========================================\n")
    for arg, content in args.__dict__.items():
        log_str += ("{}:{}\n".format(arg, content))
    log_str += ("\n==========================================\n")
    print(log_str)

def train(args):

    dsets = {}
    dsets["source"] = data_list.ImageList(open(args.s_dset_path).readlines(), transform=image_train())   # open(path).readlines()返回列表，每个元素是文件中的一行内容
    dsets["target"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_train())   # 目标域
    dsets["test"] = data_list.ImageList(open(args.test_dset_path).readlines(), transform=image_test())   # 测试集 

    dset_loaders = {}   
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size= args.batch_size, shuffle=True, num_workers=args.worker, drop_last=True, pin_memory=True, persistent_workers= True, prefetch_factor=4)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size= args.batch_size, shuffle=True, num_workers=args.worker, drop_last=True, pin_memory=True, persistent_workers= True, prefetch_factor=4)
    dset_loaders["test"]   = DataLoader(dsets["test"], batch_size= args.batch_size*2, shuffle=False, num_workers=args.worker, pin_memory=True, persistent_workers= True, prefetch_factor=4)

    base_network = en_network.ResNetFc_test(resnet_name='ResNet50', class_num=args.class_num)   # **表示字典解包操作
    base_network = base_network.cuda()      # 将模型迁移到GPU上

    parameter_list = base_network.get_parameters() 
    ## set optimizer  -- 优化器
    optimizer_config = {"type":torch.optim.SGD, 
                        "optim_params":{'lr':args.lr, "momentum":0.9, "weight_decay":5e-4, "nesterov":True}, 
                        "lr_type":"inv",
                        "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75}
    }
    optimizer = optimizer_config["type"](parameter_list,**(optimizer_config["optim_params"]))   # 定义了一个SGD优化器

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    iter_source = iter(dset_loaders["source"])   # 源域迭代器初始化
    iter_target = iter(dset_loaders["target"])   # 目标域迭代器初始化
    max_len = max(len(dset_loaders["source"]), len(dset_loaders["target"]))
    max_iter = args.max_epoch * max_len
    test_interval = 100
    best_acc = 0.0

    for i in tqdm(range(max_iter + 1), desc="Running Iterations", unit="iter"): 
        base_network.train(True)

        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if (i % test_interval == 0 and i > 0) or (i == max_iter):    
                                                
            base_network.train(False)                                                   # obtain the class-level weight and evalute the current model   测试当前训练模型的准确率
            temp_acc, mean_ent = image_test_classification(dset_loaders, base_network)  # 计算当前模型的准确率,输入字典，只用到dset_loaders["test"]中的数据
            best_acc = max(best_acc, temp_acc)
                                                                                                         
        try:
            inputs_source, labels_source, _ = next(iter_source)         # next()   每次调用next() 都会使迭代器内部指针向后移动一个批次
        except StopIteration:                                           # labels_source: 一串长度为batch_size的序列，里面的每个元素值是每一个样本的标签  
            iter_source = iter(dset_loaders["source"])
            inputs_source, labels_source, _ = next(iter_source)
        try:
            inputs_target, _, _ = next(iter_target)
        except StopIteration:
            iter_target = iter(dset_loaders["target"])
            inputs_target, _, _ = next(iter_target)                                                                
                                                                
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()  # 作为网络训练的输入
        _, outputs_source= base_network(inputs_source)  # 源域数据输入骨干网络后的输出
        _, outputs_target= base_network(inputs_target)  

        '''
        分类器的损失函数src_loss————专门用于分类器降低源域预测结果的交叉熵值 
        '''
        src_loss = torch.nn.CrossEntropyLoss()(outputs_source, labels_source)  


        softmax_tar_out = torch.nn.Softmax(dim=1)(outputs_target)
        tar_loss = torch.mean(my_loss.Entropy(softmax_tar_out))   

        total_loss = src_loss + tar_loss * 0.1



        if (i % test_interval == 0 and i > 0) or (i == max_iter):
            print_line()
     
        total_loss.backward()  
        optimizer.step()  

    return best_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='BA3US for Partial Domain Adaptation')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")

    parser.add_argument('--output', type=str, default='run')
    parser.add_argument('--seed', type=int, default=2019, help="random seed")
    parser.add_argument('--max_epoch', type=int, default=50, help="max epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=8, help="number of workers") 
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
    
    args.class_num = 10  

    # args.k = k     # 这个k表示 目标域的标签域中的 类别数量
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.s_dset_path = r'D:\LJ\workstation\Vscode\New_ISRA\data\LongSig_10\GASF_old\fixed_4class\SNR_20_4_list.txt'
    args.t_dset_path = args.s_dset_path
    args.test_dset_path =  args.t_dset_path

    os.makedirs(args.output, exist_ok=True)
    log_name = "test_output_{}.txt".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_path = os.path.join(args.output, log_name)

    with open(log_path, "w", encoding="utf-8") as log_file:
        with redirect_stdout(log_file):
            print_args(args)
            train(args)
            print("log saved in: {}".format(log_path))
