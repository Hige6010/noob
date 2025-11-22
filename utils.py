# Author: Xiaoli Wang
# Email: xiaoliw1995@gmail.com
# @Time 2024/3/28
import random
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from numpy.random import randint
import torch.nn.functional as F

#用于统计指标的平均值
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# NOTE: for incomplete data, because need to add Sn in dataset
class partial_mv_dataset(Dataset):
    def __init__(self, data, Sn, Y):
        '''
        Construct dataset according input values
        :param data: Input data is a list of numpy arrays
        '''
        self.data = data
        self.Y = Y
        self.Sn = Sn

    def __getitem__(self, item):
        #对每个视图view，提取第item个样本的特征self.data[view][item]，并通过np.newaxis扩展维度（从(feature_dim, )变为(1, feature_dim)），最终datum是一个列表，每个元素是单个样本在对应视图上的特征（形状为(1, feature_dim)）
        datum = [self.data[view][item][np.newaxis, :] for view in range(len(self.data))]
        Y = self.Y[item]
        #Sn = self.Sn[item].reshape(1, len(self.Sn[item]))：提取第 item 个样本的视图缺失掩码（Sn 是形状为 (n_views,) 的数组，reshape 后变为 (1, n_views)，便于后续广播运算）
        Sn = self.Sn[item].reshape(1, len(self.Sn[item]))
        # print((datum[0]*Sn[0]).shape)
        # NOTE: mask missing view   # * Sn[0][view]
        #通过 datum[view] * Sn[0][view] 对缺失的视图特征进行屏蔽（缺失视图的特征会被置 0）
        #Sn[0][view] 表示第 view 个视图是否缺失，1 表示存在，0 表示缺失
        return [torch.Tensor(datum[view] * Sn[0][view]) for view in range(len(self.data))], torch.Tensor(Sn), torch.Tensor(Y)

    def __len__(self):
        return self.data[0].shape[0]
#加载Sn
class partial_sn_dataset(Dataset):
    def __init__(self, Sn):
        self.Sn = Sn

    def __getitem__(self, item):

        Sn = self.Sn[item].reshape(1, len(self.Sn[item]))#将缺失标记调整为形状为 (1, 特征数) 的张量，保证数据维度的一致性。
        return torch.Tensor(Sn)

    def __len__(self):
        return self.Sn.shape[0]

class mv_dataset(Dataset):
    def __init__(self, data, Y):
        '''
        Construct dataset according input values
        :param data: Input data is a list of numpy arrays
        '''
        self.data = data
        self.Y = Y

    def __getitem__(self, item):
        #同时处理多视图特征、缺失掩码、标签。
        #最终 new_batch 会有 n_views 个元素（n_views 是视图数量），每个元素是一个空列表，用于后续按 “视图维度” 拼接特征。
        datum = [self.data[view][item][np.newaxis, :] for view in range(len(self.data))]
        Y = self.Y[item]
        #将特征和标签从 NumPy 数组转换为 PyTorch 张量，供模型训练时使用。
        return [torch.from_numpy(datum[view]) for view in range(len(self.data))], torch.from_numpy(Y)

    def __len__(self):
        return self.data[0].shape[0]

# NOTE: construct batch data for incomplete data
def partial_mv_tabular_collate(batch):
    new_batch = [[] for _ in range(len(batch[0][0]))]
    new_label = []
    new_Sn = []
    for y in range(len(batch)):
        cur_data = batch[y][0]#多视图特征
        Sn_data = batch[y][1]#视图缺失掩码
        label_data = batch[y][2]#提取第 y 个样本的标签。
        for x in range(len(batch[0][0])):
            new_batch[x].append(cur_data[x])#整个批次中所有样本的第 x 个视图的特征。
        new_Sn.append(Sn_data)
        new_label.append(label_data)
    return [torch.cat(new_batch[i], dim=0) for i in range(len(batch[0][0]))], torch.cat(new_Sn, dim=0), torch.cat(new_label, dim=0)
def partial_sn_tabular_collate(batch):
    new_Sn = []
    #仅专注于缺失掩码的批次拼接。
    for y in range(len(batch)):
        Sn_data = batch[y]
        new_Sn.append(Sn_data)
    return torch.cat(new_Sn, dim=0)

def mv_tabular_collate(batch):
    #多视图特征和标签
    new_batch = [[] for _ in range(len(batch[0][0]))]
    new_label = []
    for y in range(len(batch)):
        cur_data = batch[y][0]
        label_data = batch[y][1]
        for x in range(len(batch[0][0])):
            new_batch[x].append(cur_data[x])
        new_label.append(label_data)
    return [torch.cat(new_batch[i], dim=0) for i in range(len(batch[0][0]))],  torch.cat(new_label, dim=0)

def tensor_intersection(x, y):
    #找交集
    return torch.tensor(list(set(x.tolist()).intersection(set(y.tolist()))))

def late_fusion(input, target, sn, current_epoch, args):
    """
    :param input: list, include all view's mapping output  [view_num, B, class_num]
    :param target: gt
    :param sn: missing value indicator matrix; [B, view_num]
    :param eta: --float--ratio of ood samples
    :param current_epoch:
    :return:
    """
    batch_num, class_num = input.shape
    logits = input# 模型输出的未归一化概率（logits）

    # NOTE:
    with torch.no_grad():
        # ce weights
        probabilities = F.softmax(logits, dim=-1)# 对 logits 做 softmax，得到类别概率
        max_entropy = torch.log2(torch.tensor(class_num))# 最大熵（均匀分布时的熵）
        # 计算每个样本的熵，并转换为权重
        w = -torch.sum(probabilities * torch.log2(probabilities + 1e-5), dim=1)
        w = w / max_entropy
        w = torch.exp(-w)# 指数变换，将“高熵（难分）”样本的权重放大

    ood_num = int(args.eta * batch_num)
    if (ood_num == 0 or current_epoch < args.ours_start_step):
        # return [F.cross_entropy(predict, gt)] # unstable
        # return [F.nll_loss(logits, target, reduction='mean')]
        # return [F.cross_entropy(logits, target, reduction='mean') + entropy_loss]
        return [F.cross_entropy(logits, target, reduction='mean')]

    # first term: cross entropy loss
    # ce_loss = F.nll_loss(logits, target, reduction='none')
    #计算所有样本的交叉熵损失（reduction='none' 表示保留每个样本的损失值）
    ce_loss = F.cross_entropy(logits, target, reduction='none')

    # rk = torch.argsort((entropy+ce_loss).squeeze(-1), descending=True)
    #对交叉熵损失排序，选出“损失最大（最难分）”的 ood_num 个样本
    rk = torch.argsort((ce_loss).squeeze(-1), descending=True)

    results = []
    #计算普通样本的交叉熵损失总和
    ce_loss = torch.sum(ce_loss[rk[ood_num:]])
    # second term: kl divergence loss
    log_probs = F.log_softmax(logits[rk[:ood_num]], dim=-1)# 难分样本的 log_softmax 输出
    # weight = entropy[rk[:ood_num]] / torch.log(torch.tensor(class_num))
    target_distribution = torch.tensor([[1.0 / class_num] * class_num]).expand_as(log_probs).cuda()# 均匀分布的目标
    kl_loss = F.kl_div(log_probs, target_distribution, reduction='sum')# KL 散度损失
    loss = (ce_loss + args.beta * kl_loss) / batch_num
    # loss = (w * F.cross_entropy(logits, target)).mean()
    results.append(loss)

    return results

def setup_seed(seed):
    #  下面两个常规设置了，用来np和random的话要设置
    np.random.seed(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU训练需要设置这个
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)  # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
    torch.backends.cudnn.benchmark = False  #

def data_write_csv(filepath, datas):
    file = open(filepath, 'a+')
    file.write(datas)
    file.write('\n')

    file.close()
