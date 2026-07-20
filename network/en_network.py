import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision.models.vgg import VGG16_BN_Weights, VGG19_BN_Weights, VGG16_Weights
from .repvgg import create_RepVGG_B1g2
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):    # 一个Sigmod函数的线性变化，用于生成一个从0~1的系数
    return np.float64(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):                    # 权重初始化函数 不同网络层初始化的不同方式，有助于快速收敛
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:  # 卷积层
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:   # 批量归一化层
        nn.init.normal_(m.weight, 1.0, 0.02) 
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:      # 全连接层
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

# 定义一个字典，将 ResNet 模型名称映射到对应的权重枚举
resnet_weights_dict = {
  "ResNet18": ResNet18_Weights.DEFAULT,
  "ResNet34": ResNet34_Weights.DEFAULT,
  "ResNet50": ResNet50_Weights.DEFAULT,
  "ResNet101": ResNet101_Weights.DEFAULT,
  "ResNet152": ResNet152_Weights.DEFAULT
}

resnet_dict = {
  "ResNet18":models.resnet18, 
  "ResNet34":models.resnet34, 
  "ResNet50":models.resnet50, 
  "ResNet101":models.resnet101, 
  "ResNet152":models.resnet152
}

class ResNetFc(nn.Module):
  def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=512, embedding_dim=256):
    super(ResNetFc, self).__init__()
    model_resnet = resnet_dict[resnet_name](weights=resnet_weights_dict[resnet_name])   # 预训练模型参数  

    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1  
    self.relu = model_resnet.relu 
    self.maxpool = model_resnet.maxpool

    self.layer1 = model_resnet.layer1    # 每一个layer都是一个残差块组
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool    #全局平均池化层将特征图压缩为一维向量
    
    self.feature_layers = nn.Sequential(
    self.conv1, self.bn1, self.relu,          
    self.maxpool, 
    self.layer1, self.layer2, self.layer3, self.layer4,               
    self.avgpool                         
    )                             

    '''
    class2vec 只有1层隐藏层的前馈神经网络
    '''
    # encoding part of the auto-encoder
    self.encoder = nn.Sequential(
        nn.Linear(bottleneck_dim, embedding_dim),    # bottleneck 
        # nn.Linear(bottleneck_dim, embedding_dim),
        nn.ReLU())
    # decoding part of the auto-encoder
    self.decoder = nn.Sequential( 
        nn.Linear(embedding_dim, bottleneck_dim), 
        nn.ReLU())
       
    # self.embedding_dim = embedding_dim
    self.use_bottleneck = use_bottleneck
    if self.use_bottleneck:     # 论文使用瓶颈层：将特征维度压缩到bottleneck_dim
        self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)    # model_resnet.fc.in_features表示平均池化avgpool后的特征：2048

        self.bottleneck.apply(init_weights)              # 对新的网络层进行初始化，其余层使用预训练的参数
        self.__in_features = bottleneck_dim
  
  '''
  ResNet50_feature_layers   --> 瓶颈层 bottleneck + fc
                            --> bottleneck + class2vec     
  '''
  def forward(self, x):
    x = self.feature_layers(x)       
    x = x.view(x.size(0), -1)    
    if self.use_bottleneck: 
        x = self.bottleneck(x)
        e = self.encoder(x)   # 编码输出，维度（ batch_size, embedding_dim ）  e- encoder
        r = self.decoder(e)   # 解码输出，维度（ batch_size, bottleneck_dim ） r- decoder
    return x, e, r

  def output_num(self):
    return self.__in_features   # 指输出的X特征维度，而不是隐式语义特征维度
  @property
  def bottleneck_dim(self):
    return self.__in_features
  @property
  def embedding_dim(self):
    return self.encoder[0].out_features           

  def get_parameters(self):
    if self.new_cls:
        if self.use_bottleneck:
            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.encoder.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.decoder.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            ]
    return parameter_list


class ResNetFc_test(nn.Module):
  def __init__(self, resnet_name,  class_num=50):
    super(ResNetFc_test, self).__init__()
    model_resnet = resnet_dict[resnet_name](weights=resnet_weights_dict[resnet_name])   # 预训练模型参数  

    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1  
    self.relu = model_resnet.relu 

    self.maxpool = model_resnet.maxpool

    self.layer1 = model_resnet.layer1    # 每一个layer都是一个残差块组
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool    #全局平均池化层将特征图压缩为一维向量
    
    self.feature_layers = nn.Sequential(
    self.conv1, self.bn1, self.relu,          
    self.maxpool, 
    self.layer1, self.layer2, self.layer3, self.layer4,               
    self.avgpool                         
    )                             
    self.fc = nn.Linear(model_resnet.fc.in_features, class_num)   # 全连接层fc 用于进行最后一步分类--输出的是得分，不是概率，因为没有进行激活函数转换        

  def forward(self, x):
    x = self.feature_layers(x)       
    x = x.view(x.size(0), -1)    
    logits = self.fc(x)     # 分类结果 维度：（batch_size, class_num）
    return x, logits
  
  def get_parameters(self):
    parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                      {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
    return parameter_list


def RepVGG_B1g2_Weight():  # 返回一个含模型参数的字典
    repvgg_weight = torch.load(r'D:\LJ\workstation\Vscode\New_ISRA\pre_train\RepVGG-B1g2-train.pth')

    return repvgg_weight    

repvgg_weight_dict = {
    "RepVGG_B1g2": RepVGG_B1g2_Weight()
}

repvgg_dict = {
    "RepVGG_B1g2": create_RepVGG_B1g2()
}

def load_model_repvgg_B1g2_weights():

    model_repvgg = create_RepVGG_B1g2(deploy=False, use_checkpoint=False)
    weight_repvgg = torch.load(r'D:\LJ\workstation\Vscode\New_ISRA\pre_train\RepVGG-B1g2-train.pth')
    model_repvgg.load_state_dict(weight_repvgg, strict=True) 
    return model_repvgg

class RepVGG_B1g2(nn.Module):
    def __init__(self, use_bottleneck=True, bottleneck_dim=512, embedding_dim=256, use_checkpoint=False, pretrain_path=None):
        super (RepVGG_B1g2, self).__init__()
        if pretrain_path:
            model_repvgg = torch.load(pretrain_path)
        else:
            model_repvgg = load_model_repvgg_B1g2_weights()
        self.stage0 = model_repvgg.stage0
        self.stage1 = model_repvgg.stage1
        self.stage2 = model_repvgg.stage2
        self.stage3 = model_repvgg.stage3
        self.stage4 = model_repvgg.stage4
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.use_checkpoint = use_checkpoint

        # encoding part of the auto-encoder
        self.encoder = nn.Sequential(
            nn.Linear(bottleneck_dim, embedding_dim),    # bottleneck 
            # nn.Linear(bottleneck_dim, embedding_dim),
            nn.ReLU())
        # decoding part of the auto-encoder
        self.decoder = nn.Sequential( 
            nn.Linear(embedding_dim, bottleneck_dim), 
            nn.ReLU())
        
        self.use_bottleneck = use_bottleneck

        self.bottleneck = nn.Linear(model_repvgg.linear.in_features, bottleneck_dim)    # model_resnet.fc.in_features表示平均池化avgpool后的特征：2048

        self.bottleneck.apply(init_weights)              
        self.__in_features = bottleneck_dim 

    def forward(self, x):
        out = self.stage0(x)
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            for block in stage:
                if self.use_checkpoint:
                    out = checkpoint.checkpoint(block, out)
                else:
                    out = block(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)

        if self.use_bottleneck: 
            x = self.bottleneck(out)
            e = self.encoder(x)   
            r = self.decoder(e)  

        return x, e, r

    def output_num(self):
        return self.__in_features   
    @property
    def bottleneck_dim(self):
        return self.__in_features
    @property
    def embedding_dim(self):
        return self.encoder[0].out_features
                
    def get_parameters(self):
        parameter_list = [
                        {"params":self.stage0.parameters(), "lr_mult":1, 'decay_mult':2}, \
                        {"params":self.stage1.parameters(), "lr_mult":1, 'decay_mult':2}, \
                        {"params":self.stage2.parameters(), "lr_mult":1, 'decay_mult':2}, \
                        {"params":self.stage3.parameters(), "lr_mult":1, 'decay_mult':2}, \
                        {"params":self.stage4.parameters(), "lr_mult":1, 'decay_mult':2}, \
                        {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                        {"params":self.encoder.parameters(), "lr_mult":10, 'decay_mult':2}, \
                        {"params":self.decoder.parameters(), "lr_mult":10, 'decay_mult':2}
                        ]
        
        return parameter_list 


def load_model_repvgg_L2pse_weights():

    model_repvgg = create_RepVGG_B1g2(deploy=False, use_checkpoint=False)
    weight_repvgg = torch.load(r'D:\LJ\workstation\Vscode\New_ISRA\pre_train\RepVGGplus-L2pse-train256-acc84.06.pth')
    model_repvgg.load_state_dict(weight_repvgg, strict=True)
    return model_repvgg

class RepVGG_L2pse(nn.Module):
    def __init__(self, use_bottleneck=True, bottleneck_dim=512, new_cls=False, class_num=1000, embedding_dim=256, use_checkpoint=False):
        super (RepVGG_L2pse, self).__init__()
        model_repvgg = load_model_repvgg_L2pse_weights()
        self.stage0 = model_repvgg.stage0
        self.stage1 = model_repvgg.stage1
        self.stage2 = model_repvgg.stage2
        self.stage3 = model_repvgg.stage3
        self.stage4 = model_repvgg.stage4
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.use_checkpoint = use_checkpoint

        # encoding part of the auto-encoder
        self.encoder = nn.Sequential(
            nn.Linear(bottleneck_dim, embedding_dim),    # bottleneck 
            # nn.Linear(bottleneck_dim, embedding_dim),
            nn.ReLU())
        # decoding part of the auto-encoder
        self.decoder = nn.Sequential( 
            nn.Linear(embedding_dim, bottleneck_dim), 
            nn.ReLU())
        
        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls

        self.bottleneck = nn.Linear(model_repvgg.linear.in_features, bottleneck_dim)    # model_resnet.fc.in_features表示平均池化avgpool后的特征：2048
        self.fc = nn.Linear(bottleneck_dim, class_num)   


        self.bottleneck.apply(init_weights)              
        self.fc.apply(init_weights)
        self.__in_features = bottleneck_dim 


    def forward(self, x):
        out = self.stage0(x)
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            for block in stage:
                if self.use_checkpoint:
                    out = checkpoint.checkpoint(block, out)
                else:
                    out = block(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)

        if self.use_bottleneck and self.new_cls: 
            x = self.bottleneck(out)
            e = self.encoder(x)   
            r = self.decoder(e)  

        y = self.fc(x)     # 分类结果 维度：（batch_size, class_num）
        return x, y, e, r


    def output_num(self):
        return self.__in_features   
    @property
    def bottleneck_dim(self):
        return self.__in_features
    @property
    def embedding_dim(self):
        return self.encoder[0].out_features
                

    def get_parameters(self):
        parameter_list = [
                        {"params":self.stage0.parameters(), "lr_mult":1, 'decay_mult':2}, \
                        {"params":self.stage1.parameters(), "lr_mult":1, 'decay_mult':2}, \
                        {"params":self.stage2.parameters(), "lr_mult":1, 'decay_mult':2}, \
                        {"params":self.stage3.parameters(), "lr_mult":1, 'decay_mult':2}, \
                        {"params":self.stage4.parameters(), "lr_mult":1, 'decay_mult':2}, \
                        {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                        {"params":self.encoder.parameters(), "lr_mult":10, 'decay_mult':2}, \
                        {"params":self.decoder.parameters(), "lr_mult":10, 'decay_mult':2}, \
                        {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
        
        return parameter_list 

vgg_dict = {
            "VGG16":models.vgg16, 
            "VGG16BN":models.vgg16_bn, 
            "VGG19BN":models.vgg19_bn
} 
# 定义一个字典，将模型名称映射到对应的权重枚举
vgg_weights_dict = {
            "VGG16":VGG16_Weights.DEFAULT,
            "VGG16_BN": VGG16_BN_Weights.DEFAULT, 
            "VGG19_BN": VGG19_BN_Weights.DEFAULT, 
}


class VGGFc(nn.Module):
  def __init__(self, vgg_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, embedding_dim=128):
    super(VGGFc, self).__init__()
    model_vgg = vgg_dict[vgg_name](weights=vgg_weights_dict[vgg_name])
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.feature_layers = nn.Sequential(self.features, self.classifier)

    # encoding part of the auto-encoder
    self.encoder = nn.Sequential(
        nn.Linear(bottleneck_dim, embedding_dim),
        # nn.Linear(bottleneck_dim, embedding_dim),
        nn.ReLU())
    # decoding part of the auto-encoder
    self.decoder = nn.Sequential( 
        nn.Linear(embedding_dim, bottleneck_dim), 
        nn.ReLU())

    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(4096, bottleneck_dim)
    
            self.bottleneck.apply(init_weights)

            self.__in_features = bottleneck_dim


  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    if self.use_bottleneck and self.new_cls:
        x = self.bottleneck(x)
        e = self.encoder(x)
        r = self.decoder(e)

    return x, e, r

  def output_num(self):
    return self.__in_features
  @property
  def bottleneck_dim(self):
    return self.__in_features
  @property
  def embedding_dim(self):
    return self.encoder[0].out_features

  def get_parameters(self):
    if self.new_cls:
        if self.use_bottleneck:
            parameter_list = [{"params":self.features.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.classifier.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.encoder.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.decoder.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            ]
        else:
            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.classifier.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            ]
    else:
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
    return parameter_list


class AdversarialNetwork(nn.Module):     # DANN域判别器--三层隐层
  def __init__(self, in_feature, hidden_size, max_iter = 10000):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)    # 随机将神经元输出为0，提高模型泛化能力
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = max_iter

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1

  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]


class MLP_regressor(torch.nn.Module):    # 单隐层的多层感知机模型
    def __init__(self, n_feature, n_hidden_1): 
        super(MLP_regressor, self).__init__()
        self.hidden_1 = torch.nn.Linear(n_feature, n_hidden_1)
        self.predict = torch.nn.Linear(n_hidden_1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden_1(x))   # 把F.relu() 改成了 nn.Relu()
        x = self.predict(x)
        return x

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult":1, 'decay_mult':10}]


class MLPS_regressores(torch.nn.Module):    # 多个MLP回归器
    def __init__(self, num_mlps, n_feature, n_hidden_1):
        super(MLPS_regressores, self).__init__()
        self.mlps = torch.nn.ModuleList(
            [MLP_regressor(n_feature, n_hidden_1) for _ in range(num_mlps)]
        )

    def append(self, mlp):
        self.mlps.append(mlp)

    def __len__(self):
        return len(self.mlps)

    def __getitem__(self, index):
        return self.mlps[index]

    def __iter__(self):
        return iter(self.mlps)

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult":1, 'decay_mult':10}]

class ContrastiveProjectionHead(nn.Module):
    """Two-layer projection head used only by contrastive pretraining."""

    def __init__(self, input_dim, hidden_dim=None, output_dim=128):
        super(ContrastiveProjectionHead, self).__init__()
        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.layers(x)

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult":10, 'decay_mult':2}]


class StochasticClassifier(nn.Module):
    '''
    StochasticClassifier
    '''
    def __init__(self, feature_dim, num_classes):
        super(StochasticClassifier, self).__init__()
        self.mean = nn.Linear(feature_dim , num_classes) # 平均权重 高斯分布的均值μ
        self.std = nn.Linear(feature_dim , num_classes)  # 扰动幅度 高斯分布的方差σ
        
        self.apply(init_weights)

    def forward(self, x, mode='train'):
        if mode == 'test':
            weight = self.mean.weight
            bias = self.mean.bias
        else:            
            e_weight = torch.randn_like(self.mean.weight)
            e_bias = torch.randn_like(self.mean.bias)
            
            # 随机分类器权重 = 平均权重 + 扰动幅度 * 随机噪声 
            weight = self.mean.weight + self.std.weight * e_weight
            bias = self.mean.bias + self.std.bias * e_bias

        # 等价于 F.linear(x, weight, bias) --> logits
        out = torch.matmul(x, weight.t()) + bias
        return out
    
    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult":10, 'decay_mult':2}]
    

    '''
    torch.randn( torch.size([30,512]) )    
        表示生成一个[30,512] 性质的随机张量, 其中的每一个元素都来自标准正态分布N(0,1)

    '''


class common_fc(nn.Module):
    def __init__(self, in_features, num_classes):
        super(common_fc, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.apply(init_weights)

    def forward(self, x):
        return self.fc(x)

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult":10, 'decay_mult':2}]
