import argparse
import os, random, pdb, math, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import utils.my_loss as my_loss 
import network.en_network as en_network
import utils.lr_schedule as lr_schedule
import utils.data_list as data_list
import torch.nn.functional as F
#from sklearn.metrics import confusion_matrix
import pandas as pd
from tqdm import tqdm
import torch.utils.checkpoint as checkpoint
from torch.utils.data import Subset
import copy
from datetime import datetime

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

def image_test_classification(loader, base_model, classifiers):
    start_test = True
    
    with torch.no_grad():  # 下述代码不进行梯度计算

        iter_test = iter(loader["test"])       # 测试集中batch_size*2 
        total_samples_iter = 0
        for i in range(len(loader['test'])):   # 会遍历一个epoch全部目标域都测试一遍   
            data = next(iter_test)             # iter_test.next()返回一个batch的数据
            inputs = data[0]
            labels = data[1]
            
            batch_size_flag = inputs.shape[0]   # 用于查验test的样本数目
            total_samples_iter += batch_size_flag

            inputs = inputs.cuda()  # 
            out_f,  _, _ = base_model(inputs) # 模型输出特征
            outputs = classifiers(out_f, mode='test')

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()   #
                start_test = False   
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0) # 每一次调用image_classification，都会遍历所有的测试集， 
                                                                               # 相当于把目标域给完整的遍历了一遍，理论上只应该对1/10的目标域进行有标签的测试才行  
                                                                               # 其实不是的，就是要遍历整个目标域，来 
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)    # .max()选择了其中最大概率输出的类别标签  
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])   # 准确率

    return accuracy 


def get_Classifier_logits_probs(Classifiers, feat):
    logits = Classifiers(feat)
    probs = torch.nn.Softmax(dim=1)(logits)
    return logits, probs

def savemodel(best_model, args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    snr = (args.t-1)*5
    file_name = f"TD_{args.multi_class}class_SNR_{snr}_best_model_en_{timestamp}.pt"
    save_dir = os.path.join('model',f'{args.net}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, file_name)
    try:
        torch.save(best_model, model_path)
        print(f"model is saved in: {model_path}")
    except Exception as e:
        print(f"model is not saved: {e}")


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
    ## prepare data    测试集的batch_size时训练集batch_size的两倍
    train_bs, test_bs = args.batch_size, args.batch_size * 2
    '''
    读取数据
    '''
    dsets = {}
    dsets["source"] = data_list.ImageList(open(args.s_dset_path).readlines(), transform=image_train())   # open(path).readlines()返回列表，每个元素是文件中的一行内容
    dsets["test"] = data_list.ImageList(open(args.test_dset_path).readlines(), transform=image_test())   # 测试集 

    '''
    dset_loader数据加载器字典
    '''
    dset_loaders = {}   
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True, pin_memory=True, persistent_workers= True)
    dset_loaders["test"]   = DataLoader(dsets["test"], batch_size=test_bs, shuffle=False, num_workers=args.worker, pin_memory=True, persistent_workers= True)

    max_len = len(dset_loaders["source"])
    args.max_iter = args.max_epoch * max_len
    args.test_interval = int(args.max_iter / 20) 
    print("max_len:", max_len)
    print("args.max_iter: ", args.max_iter)
    print("args.test_interval: ", args.test_interval)
    print()

    if "ResNet50" in args.net:
        params = {"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim": args.fdim, "new_cls":True, "embedding_dim": args.edim}
        base_network = en_network.ResNetFc(**params)   # **表示字典解包操作

    if "RepVGG_B1g2" in args.net:
        params = {"use_bottleneck":True, "bottleneck_dim":args.fdim, "new_cls":True, "embedding_dim": args.edim}
        base_network = en_network.RepVGG_B1g2(**params)

    base_network = base_network.cuda()      # 将模型迁移到GPU上

    C= en_network.StochasticClassifier(base_network.output_num(), args.class_num).cuda()

    parameter_list = base_network.get_parameters() + C.get_parameters()

    ## optimizer setting 
    optimizer_config = {
        "type":torch.optim.SGD, 
        "optim_params":{
            'lr':args.lr, 
            "momentum":0.9, 
            "weight_decay":5e-4, 
            "nesterov":True
        }, 
        "lr_type":"inv",
        "lr_param":{
            "lr":args.lr, 
            "gamma":0.001, 
            "power":0.75
        }
    }
    ## optimizer create
    optimizer = optimizer_config["type"](
        parameter_list,
        **(optimizer_config["optim_params"])
    )  

    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    class_weight = None
    best_ent = 1000
    best_acc = 0
    should_stop = False
    no_improve_epochs = 0
    iter_source = iter(dset_loaders["source"])   # 源域迭代器初始化
    iter_target = iter(dset_loaders["target"])   # 目标域迭代器初始化

    for i in tqdm(range(args.max_iter + 1), 
                  desc="Running Iterations", 
                  dynamic_ncols=True,
                  unit="iter"): 
        
        base_network.train(True)
        C.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)


        if (i % args.test_interval == 0 and i > 0) or (i == args.max_iter):    
                                                
            base_network.train(False) 
            C.train(False)                                                                # obtain the class-level weight and evalute the current model   测试当前训练模型的准确率
            temp_acc = image_test_classification(dset_loaders, base_network, C)  # 计算当前模型的准确率,输入字典，只用到dset_loaders["test"]中的数据
            base_network.train(True)
            C.train(True)
            class_weight = class_weight.cuda().detach()                                               # 复制预测的类级权重class_weight-（test data中各类别的占比（会随着迭代的增加而变化）)
                                                            
            if  best_acc < temp_acc:
                best_acc = temp_acc                                         # 若当前分类的平均熵 低于历史最低平均，那么进行替换
                best_model = {
                    "base_network":copy.deepcopy(base_network.state_dict()),                    # 将当前模型base_network的所有网络参数保存 
                    "classifier":copy.deepcopy(C.state_dict()),
                    "best_acc": best_acc,
                    "net": args.net,
                    "class_num": args.class_num,
                    "classifier_num": args.classifiers_num
                    }                             
                no_improve_epochs = 0 
            else:                                                                              # early stop
                no_improve_epochs += 1*(args.max_epoch / 20) 
                if no_improve_epochs >= args.early_stop: 
                    should_stop = True 

            if should_stop:
                print("Train stopped early due to no improvement in mean entropy in continuous {} epochs".format(args.early_stop)) 
                break  
            current_epoch = i // max_len
            print("current epoch: ", current_epoch)  
            print("iter: {:05d}, precision: {:.5f}, best_acc:{:.5f}".format(i, temp_acc, best_acc))   
                                
        try:
            inputs_source, labels_source, _ = next(iter_source)         # next()   每次调用next() 都会使迭代器内部指针向后移动一个批次
        except StopIteration:                                           # labels_source: 一串长度为batch_size的序列，里面的每个元素值是每一个样本的标签  
            iter_source = iter(dset_loaders["source"])
            inputs_source, labels_source, _ = next(iter_source)
                                                             
                                                                
        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()  # 作为网络训练的输入

        if class_weight is not None and class_weight[labels_source].sum() == 0:
            continue

        fea_s, e_s, r_s = base_network(inputs_source)  # 源域数据输入base网络后的输出

        features = fea_s.clone()

        '''
        Stochastic classifier avg prediction

        源域交叉熵损失*类级权重
        '''
        probs_sum = 0.0
        src_loss_sum = 0.0
        for _ in range(args.classifiers_num):        

            logits, probs = get_Classifier_logits_probs(C, features)
            probs_sum += probs

            if class_weight is not None: 
                src_ = torch.nn.CrossEntropyLoss(reduction='none')(logits[:fea_s.size(0)], labels_source)  # reduction 控制是否自动计算所有样本损失的平均
                weight = class_weight[labels_source].detach()
                src_loss = torch.sum(weight * src_) / (1e-8 + torch.sum(weight).item())    # 交叉熵损失进行加权+平均
            else:
                src_loss = torch.nn.CrossEntropyLoss()(logits[:fea_s.size(0)], labels_source)     # 在第一次test之前，使用标准交叉熵损失
            # every classifier loss summation
            src_loss_sum += src_loss
        
        src_loss = src_loss_sum / args.classifiers_num   # K 次采样交叉熵损失的平均
        probs_avg = probs_sum / args.classifiers_num     # K 次采样概率的平均
  
        
        '''
        总损失total_loss
        '''
        total_loss = src_loss 

        if (i % args.test_interval == 0 and i > 0) or (i == args.max_iter):
            print("src :", src_loss.item(), ", Trans: ")             
            print_line()
     
        total_loss.backward()  
        optimizer.step()  

    savemodel(best_model, args)

    log_str = 'Acc: ' + str(np.round(best_acc*100, 2)) + "\n" + 'Mean_ent: ' + str(np.round(best_ent, 3)) + '\n'
    print(log_str)

    return best_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='26-7-14')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--output', type=str, default='run')
    parser.add_argument('--seed', type=int, default=2026, help="random seed")
    parser.add_argument('--max_epoch', type=int, default=50, help="max epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=8, help="number of workers") 
    parser.add_argument('--net', type=str, default='ResNet50', help=["ResNet50", "VGG16", "ResNet101", "ResNet152", "RepVGG_B1g2"])
    
    parser.add_argument('--dset', type=str, default='LongSig_50')  # , choices=["office", "office_home", "imagenet_caltech"]
    parser.add_argument('--cls_type', type=str, default='fixed_30class', help="  fixed_30class, Multi_class, Train10_Test0 ")
    parser.add_argument('--multi_class', type=str, default='30', help=" Target class num = 10, 15, 20, 25, 30")   
    parser.add_argument('--K', type=int, default=5, help="Top-K") 
    parser.add_argument('--momentum', type=float, default=0.9)    
    parser.add_argument('--early_stop', type=int, default=15, help="early stop") 
    parser.add_argument('--classifiers_num', type=int, default=3, help="classifier num") 
    
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--ent_weight', type=float, default=0.1)
    parser.add_argument('--fdim', type=int, default=512) # 对应bottom_feature 
    parser.add_argument('--edim', type=int, default=256) # 对应decoder_feature 
    parser.add_argument('--rcwt', type=float, default=1.0) 
    parser.add_argument('--alwt', type=float, default=1.0) 
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id 

    '''
                 SNR_-5 : 0     SNR_0 : 1    
                 SNR_5  :2      SNR_10 : 3  
                 SNR_15 :4      SNR_20 : 5  
                 SNR_-5_RayL:6  SNR_0_RayL:7
                 SNR_5_RayL :8  SNR_10_RayL:9
                 SNR_15_RayL:10  SNR_20_RayL:11           
    '''

    if args.dset in ( r'LongSig_50', 
                     ):     
        names = ['SNR_-5', 'SNR_0', 
                 'SNR_5' , 'SNR_10',
                 'SNR_15', 'SNR_20',
                 'SNR_-5_RayL', 'SNR_0_RayL',
                 'SNR_5_RayL',  'SNR_10_RayL', 
                 'SNR_15_RayL', 'SNR_20_RayL',                 
                 ]
        args.class_num = 50   
        args.lr=1e-3

    # args.k = k     # 这个k表示 目标域的标签域中的 类别数量
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_folder = r'D:\LJ\workstation\Vscode\New_ISRA\data'
    args.s_dset_path = r'D:\LJ\workstation\Vscode\New_ISRA\data\LongSig_50\Source\GASF_old\SNR_20_list.txt'

    args.t_dset_path = data_folder + '\\'+ args.dset + '\\' + 'Target\\' + args.cls_type +'\\'+ names[args.t] + '_' + args.multi_class + '_list.txt'   # 目标域的 标签的路径
    args.test_dset_path =  args.t_dset_path

    args.name = names[args.s] +' to '+ names[args.t]  # 用来把源域和目标域的缩写拼接在一起，[0]获取字符串的第一个元素，.upper()转为大写

    print_args(args)

    train(args)  
