import argparse
import os, random, pdb, math, sys
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
    mean_ent = torch.mean(my_loss.Entropy(torch.nn.Softmax(dim=1)(all_output))).cpu().data.item()           # 平均熵 维度[类别数量]，每一个元素值表示这个测试集样本的预测的混乱程度（置信度）

    hist_tar = torch.nn.Softmax(dim=1)(all_output).sum(dim=0)   # 输入维度：all_output--[样本数量，类别数量]
                                                                # nn.Softmax(dim=1)(all_output)表示对每一个样本预测输出进行归一化---将每一个样本的每一类的预测得分转换为预测概率
                                                                # .sum(dim=0)表示 把所有样本，每一类对应下的预测概率进行求和
                                                                # 输出的维度：hist_tar--[类别数量] 
    hist_tar = hist_tar / hist_tar.sum()    # 每个类别占总类别的输出概率 （算是归一化了），用于表示一个类别在测试集中的比例
    return accuracy, hist_tar, mean_ent


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
    dset数据集字典, 接收包含图像信息的列表和图像变换操作作为参数
    内的数据格式：(tensor_image, label),
    tensor_image: 形状为 [channels, height, width] 
    label: 整数标签
    '''
    dsets = {}
    dsets["source"] = data_list.ImageList(open(args.s_dset_path).readlines(), transform=image_train())   # open(path).readlines()返回列表，每个元素是文件中的一行内容
    dsets["target"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_train())   # 目标域
    dsets["test"] = data_list.ImageList(open(args.test_dset_path).readlines(), transform=image_test())   # 测试集 

    '''
    dset_loader数据加载器字典: 从数据集字典dset{}中读取一批数据
    DataLoader 是 PyTorch 中用于批量加载数据的工具
    按 train_bs 的批次大小进行划分, 数据会被打乱顺序 (shuffle=True), 使用args.worker个工作线程进行数据加载, 并且丢弃最后一个不完整的批次 (drop_last=True) 
    '''
    dset_loaders = {}   
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True, pin_memory=True, persistent_workers= True)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True, pin_memory=True, persistent_workers= True)
    dset_loaders["test"]   = DataLoader(dsets["test"], batch_size=test_bs, shuffle=False, num_workers=args.worker, pin_memory=True, persistent_workers= True)

    max_len = max(len(dset_loaders["source"]), len(dset_loaders["target"]))
    args.max_iter = args.max_epoch * max_len
    args.test_interval = int(args.max_iter / 20) 
    print("max_len:", max_len)
    print("args.max_iter: ", args.max_iter)
    print("args.test_interval: ", args.test_interval)
    print()

    if "ResNet50" in args.net:
        params = {"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim": args.fdim, "new_cls":True, "embedding_dim": args.edim}
        # base_network = network.ResNetFc(**params)  
        base_network = en_network.ResNetFc(**params)   # **表示字典解包操作

    if "VGG16" in args.net:
        params = {"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":args.fdim, "new_cls":True, "embedding_dim": args.edim}
        base_network = en_network.VGGFc(**params)

    if "RepVGG_B1g2" in args.net:
        params = {"use_bottleneck":True, "bottleneck_dim":args.fdim, "new_cls":True, "embedding_dim": args.edim}
        base_network = en_network.RepVGG_B1g2(**params)

    if "DenseNet169" in args.net:
        params = {"dn_name":args.net, "bottleneck_dim":args.fdim,  "embedding_dim": args.edim,"class_num":args.class_num}
        base_network = densenet169.DenseNet169(**params)

    base_network = base_network.cuda()      # 将模型迁移到GPU上

    ## MLPS setting 
    emb_dim = args.edim     # get embedding dimension--隐式语义维度
    MLPS = []
    mlp_paras = []
    # MLPS的个数，就是emb_dim值
    for i in range(emb_dim):     
        MLPS.append(en_network.MLP_regressor(base_network.bottleneck_dim, 128).cuda())    # 将多个单隐层MLPS添加到列别mlps中
        mlp_paras = mlp_paras + MLPS[i].get_parameters()   # 添加单隐层MLP的网络参数 

    AD_NET = en_network.AdversarialNetwork(base_network.output_num(), 1024, args.max_iter).cuda()   # 对抗网络 
    C= en_network.StochasticClassifier(base_network.output_num(), args.class_num).cuda()

    parameter_list = base_network.get_parameters() + AD_NET.get_parameters() + mlp_paras + C.get_parameters()

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

    for i in tqdm(range(args.max_iter + 1), desc="Running Iterations", unit="iter"): 
        base_network.train(True)
        AD_NET.train(True)
        C.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)

        for mlp in MLPS:
            mlp.train(True)

        if (i % args.test_interval == 0 and i > 0) or (i == args.max_iter):    
                                                
            base_network.train(False) 
            C.train(False)                                                                # obtain the class-level weight and evalute the current model   测试当前训练模型的准确率
            temp_acc, class_weight, mean_ent = image_test_classification(dset_loaders, base_network, C)  # 计算当前模型的准确率,输入字典，只用到dset_loaders["test"]中的数据
            base_network.train(True)
            C.train(True)
            class_weight = class_weight.cuda().detach()                                               # 复制预测的类级权重class_weight-（test data中各类别的占比（会随着迭代的增加而变化）)
                                                            
            if  mean_ent < best_ent:
                best_ent, best_acc = mean_ent, temp_acc                                         # 若当前分类的平均熵 低于历史最低平均，那么进行替换
                best_model = {
                    "base_network":copy.deepcopy(base_network.state_dict()),
                    "classifier":copy.deepcopy(C.state_dict()),
                    "best_acc": best_acc,
                    "best_ent": best_ent,
                    "net": args.net,
                    "class_num": args.class_num,
                    "classifier_num": args.classifiers_num
                    }
                                                       # 将当前模型base_network的所有网络参数保存 
                no_improve_epochs = 0 
            # early stop
            else:
                no_improve_epochs += 1*(args.max_epoch / 20) 
                if no_improve_epochs >= args.early_stop: 
                    should_stop = True 

            if should_stop:
                print("Train stopped early due to no improvement in mean entropy in continuous {} epochs".format(args.early_stop)) 
                break                        

        try:
            inputs_source, labels_source, _ = next(iter_source)         # next()   每次调用next() 都会使迭代器内部指针向后移动一个批次
        except StopIteration:                                           # labels_source: 一串长度为batch_size的序列，里面的每个元素值是每一个样本的标签  
            iter_source = iter(dset_loaders["source"])
            inputs_source, labels_source, _ = next(iter_source)
        try:
            inputs_target, _, idx = next(iter_target)
        except StopIteration:
            iter_target = iter(dset_loaders["target"])
            inputs_target, _, idx = next(iter_target)                                                                
                                                                
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()  # 作为网络训练的输入

        if class_weight is not None and class_weight[labels_source].sum() == 0:
            continue


        fea_s, e_s, r_s = base_network(inputs_source)  # 源域数据输入base网络后的输出
        fea_t, e_t, r_t = base_network(inputs_target)  
        features = torch.cat((fea_s, fea_t), dim=0)



        '''
        计算自编码器class2vec的误差loss 
        '''
        embeddings = torch.cat((e_s, e_t), dim=0)
        recons = torch.cat((r_s, r_t), dim=0)
        f_dims = []
        e_dims = []
        for j in range(emb_dim):   
            f_dims.append(features.detach().clone().requires_grad_(True).cuda())   # .requires_grad_(True) 是计算注意力权重的关键设置
            e_dims.append(embeddings[:, j].detach().clone().cuda())   # 取所有样本在第 j 维上的 embedding 值 放到embeddings[j]上

        rc_loss = F.mse_loss(features, recons)   # 计算均方误差，backbone网络提取的特征X 与 编码解码后输出的X之间的重构误差

        '''
        多层感知机gj的损失函数Loss_reg 
        '''
        l1_weight = 0.5         # L1正则化权重系数
        mlp_losses = my_loss.MLP_Reg_Loss(MLPS, f_dims, e_dims, l1_weight) 
        optimizer.zero_grad()        
        mlp_losses.backward()    

        '''
        语义topic对齐
        '''
        align_losses = my_loss.Semantic_Alignment_Loss(
            features,
            f_dims,
            fea_s.size(0),
            fea_t.size(0)
        )

        cls_weight = torch.ones(features.size(0)).cuda()  # 一个全为1的张量 cls_weight
        if class_weight is not None:
            cls_weight[0:train_bs] = class_weight[labels_source]        # 因为打乱了source_的样本，所以每一个样本的对应的类级权重都是不一样的


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
        DANN损失函数transfer_loss 
        '''
        entropy = my_loss.Entropy(probs_avg)           # 计算信息熵（预测概率向量所包含信息的混乱程度）
        transfer_loss = my_loss.DANN_Loss(features, AD_NET, entropy, en_network.calc_coeff(i, 1, 0, 10, args.max_iter), cls_weight)   
        
        '''
        目标域样本的预测信息熵tar_loss
        '''
        probs_t = probs_avg[fea_s.size(0):]
        tar_loss = torch.mean(my_loss.Entropy(probs_t))   
        
        '''
        总损失total_loss
        '''
        total_loss = src_loss + tar_loss * args.ent_weight + transfer_loss + rc_loss * args.rcwt + align_losses * args.alwt 

        if (i % args.test_interval == 0 and i > 0) or (i == args.max_iter):
            print("src :", src_loss.item(), ", Trans: ", transfer_loss.item(), ", Tar: ", args.ent_weight * tar_loss.item())             
            print("rc :", rc_loss.item() * args.rcwt, ", align: ", align_losses.item() * args.alwt, "\n")
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
