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
        datum = [self.data[view][item][np.newaxis, :] for view in range(len(self.data))]
        Y = self.Y[item]
        Sn = self.Sn[item].reshape(1, len(self.Sn[item]))
        # print((datum[0]*Sn[0]).shape)
        # NOTE: mask missing view   # * Sn[0][view]
        return [torch.Tensor(datum[view] * Sn[0][view]) for view in range(len(self.data))], torch.Tensor(Sn), torch.Tensor(Y)

    def __len__(self):
        return self.data[0].shape[0]
class partial_sn_dataset(Dataset):
    def __init__(self, Sn):
        self.Sn = Sn

    def __getitem__(self, item):

        Sn = self.Sn[item].reshape(1, len(self.Sn[item]))
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
        datum = [self.data[view][item][np.newaxis, :] for view in range(len(self.data))]
        Y = self.Y[item]
        return [torch.from_numpy(datum[view]) for view in range(len(self.data))], torch.from_numpy(Y)

    def __len__(self):
        return self.data[0].shape[0]

# NOTE: construct batch data for incomplete data
def partial_mv_tabular_collate(batch):
    new_batch = [[] for _ in range(len(batch[0][0]))]
    new_label = []
    new_Sn = []
    for y in range(len(batch)):
        cur_data = batch[y][0]
        Sn_data = batch[y][1]
        label_data = batch[y][2]
        for x in range(len(batch[0][0])):
            new_batch[x].append(cur_data[x])
        new_Sn.append(Sn_data)
        new_label.append(label_data)
    return [torch.cat(new_batch[i], dim=0) for i in range(len(batch[0][0]))], torch.cat(new_Sn, dim=0), torch.cat(new_label, dim=0)
def partial_sn_tabular_collate(batch):
    new_Sn = []
    for y in range(len(batch)):
        Sn_data = batch[y]
        new_Sn.append(Sn_data)
    return torch.cat(new_Sn, dim=0)

def mv_tabular_collate(batch):
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
    logits = input

    # NOTE:
    with torch.no_grad():
        # ce weights
        probabilities = F.softmax(logits, dim=-1)
        max_entropy = torch.log2(torch.tensor(class_num))
        w = -torch.sum(probabilities * torch.log2(probabilities + 1e-5), dim=1)
        w = w / max_entropy
        w = torch.exp(-w)

    ood_num = int(args.eta * batch_num)
    if (ood_num == 0 or current_epoch < args.ours_start_step):
        # return [F.cross_entropy(predict, gt)] # unstable
        # return [F.nll_loss(logits, target, reduction='mean')]
        # return [F.cross_entropy(logits, target, reduction='mean') + entropy_loss]
        return [F.cross_entropy(logits, target, reduction='mean')]

    # first term: cross entropy loss
    # ce_loss = F.nll_loss(logits, target, reduction='none')
    ce_loss = F.cross_entropy(logits, target, reduction='none')

    # rk = torch.argsort((entropy+ce_loss).squeeze(-1), descending=True)
    rk = torch.argsort((ce_loss).squeeze(-1), descending=True)

    results = []

    ce_loss = torch.sum(ce_loss[rk[ood_num:]])
    # second term: kl divergence loss
    log_probs = F.log_softmax(logits[rk[:ood_num]], dim=-1)
    # weight = entropy[rk[:ood_num]] / torch.log(torch.tensor(class_num))
    target_distribution = torch.tensor([[1.0 / class_num] * class_num]).expand_as(log_probs).cuda()
    kl_loss = F.kl_div(log_probs, target_distribution, reduction='sum')
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