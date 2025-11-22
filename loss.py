# Author: Xiaoli Wang
# Email: xiaoliw1995@gmail.com
# @Time 2024/4/19
# !/user/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

mse = nn.MSELoss()
#基于熵加权的模型蒸馏损失
'''
fm_s	学生模型输出的融合特征（形状：(总样本数, 融合特征维度)）
fm_t	教师模型输出的融合特征（形状同上，作为 “目标特征” 用于对齐）
reduction='none'	损失函数不聚合（输出形状与fm_s一致，保留每个样本的逐特征损失）
logit_t：教师模型的原始输出（未经过 softmax 的 “对数几率”，形状为 (样本数, 类别数)）
'''
class MAD(nn.Module):
    def __init__(self):
        super(MAD, self).__init__()

    def forward(self, fm_s, fm_t, logit_t):
        #方均差损失
        loss = F.mse_loss(fm_s, fm_t, reduction='none')
        # print("loss: ", loss.shape)
        # strategy 1
        # logit_t_prob = F.softmax(logit_t, dim=1)
        # H_teacher = torch.sum(-logit_t_prob * torch.log(logit_t_prob), dim=1)
        # H_teacher_prob = H_teacher / torch.sum(H_teacher)
        # # print("Weight: ", H_teacher_prob.shape)
        # loss_result = torch.mean(loss * H_teacher_prob.unsqueeze(1))
        ## 基于教师模型的预测熵计算权重，对损失进行加权
        probabilities = F.softmax(logit_t, dim=-1)
        # 最大熵（类别数的对数）
        max_entropy = torch.log2(torch.tensor(logit_t.shape[1]))
        # 计算当前预测的熵（熵越大，样本越难分，权重越高）
        w = -torch.sum(probabilities * torch.log2(probabilities + 1e-5), dim=1)
        w = w / max_entropy
        # 加权平均损失
        loss_result = torch.mean(w.unsqueeze(1) * loss)

        return loss_result
        # return loss

    def mse_loss(self, fm_t, fm_s):
        return mse(fm_s, fm_t)

    def new_kd(self, fm_s, fm_t, logit_t, logit_s):

        loss = F.mse_loss(fm_s, fm_t, reduction='none')
        # print("loss: ", loss.shape)
        # strategy 1
        # logit_t_prob = F.softmax(logit_t, dim=1)
        # H_teacher = torch.sum(-logit_t_prob * torch.log(logit_t_prob), dim=1)
        # H_teacher_prob = H_teacher / torch.sum(H_teacher)
        # # print("Weight: ", H_teacher_prob.shape)
        # loss_result = torch.mean(loss * H_teacher_prob.unsqueeze(1))
        '''
        soft_max:将原始得分映射到 [0,1] 区间，且每行（每个样本）的和为 1，满足概率的定义。
        因为后续计算损失（如交叉熵）时，直接用 log 概率更高效、数值更稳定（避免 softmax 后极小概率导致的计算下溢）。
        '''
        _, lbs_t = torch.max(F.log_softmax(logit_t, dim=-1), dim=1)
        _, lbs_s = torch.max(F.log_softmax(logit_s, dim=-1), dim=1)
        flag = (lbs_t == lbs_s).float()# 预测一致为1，不一致为0
        # print("loss: ", loss.shape)
        # print("flag: ", flag.shape)
        result = torch.mean((1-flag).unsqueeze(1) * loss)# 仅对预测不一致的样本加权损失

        # probabilities = F.softmax(logit_t, dim=-1)
        # max_entropy = torch.log2(torch.tensor(logit_t.shape[1]))
        # w = -torch.sum(probabilities * torch.log2(probabilities + 1e-5), dim=1)
        # w = w / max_entropy
        # loss_result = torch.mean(w.unsqueeze(1) * loss)

        return result

#KL 函数：KL 散度计算（用于分布对齐）
def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()#在 GPU 上创建一个初始值全为 1 的张量，用于后续的权重、系数等场景，
    '''
    torch.sum 是 PyTorch 中用于求和的函数。
    对张量 alpha 沿着 dim=1（第 2 个维度）进行求和操作。
    keepdim=True 表示求和后保持原来的维度结构，即输出张量的维度与输入张量相比，仅 dim=1 维度的大小变为 1，其他维度大小不变。
    '''
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    # 基于Gamma函数的KL散度计算（用于Dirichlet分布之间的对齐）
    '''
    计算与 Gamma 函数相关的对数项。torch.lgamma 是对数伽马函数，用于处理 Gamma 分布相关的概率计算，这里通过对 S_alpha（聚合后的 alpha）和 alpha 分别计算对数伽马函数并做差、求和，得到分布的对数归一化项 lnB。
    '''
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)#torch.digamma 是双伽马函数（伽马函数的对数导数），计算 S_alpha 的双伽马函数值 dg0，用于后续的梯度类计算
    dg1 = torch.digamma(alpha)
    #结合前面计算的各项，通过元素级运算、求和以及对数项的组合，最终得到 KL 散度 kl。这一步整合了分布参数（alpha、beta）和伽马函数相关的导数、对数项，量化两个分布之间的差异。
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


#用于 “不确定性感知的分类任务”。它将分类损失（A）与 KL 散度正则化（B）结合，让模型在学习分类的同时，显式建模预测的不确定性。
def evidence_loss(p, alpha, c, global_step, annealing_step):
     '''
    alpha 是模型输出的 “证据张量”（形状为 (样本数, 类别数)），代表每个类别对应的证据强度。
    p 是模型预测的类别索引（形状为 (样本数,)）。
    A分类损失
    B KL正则化损失
    OOD训练模型中为训练到的样本
    '''
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1#将 alpha 每个元素减 1，得到调整后的证据张量 E。这一步是为了后续与 “狄利克雷分布” 的参数形式对齐
    label = F.one_hot(p, num_classes=c)#F.one_hot 将类别索引转换为独热编码（形状为 (样本数, 类别数)），例如类别2（c=3时）会被编码为 [0, 0, 1]。
    '''
    torch.digamma(S) - torch.digamma(alpha) 计算 “总证据的双伽马” 与 “每个类别证据的双伽马” 的差值。
    label * (...) 通过独热标签筛选 “真实类别” 对应的差值（非真实类别项会被置 0）
    实类别的差值（digamma(S) - digamma(alpha_真实类)）越小越好
    '''
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    '''
    global_step 是当前训练步数，annealing_step 是预设的 “退火步数”。
    系数用于逐渐增强 KL 散度正则化的权重：训练初期系数小（正则化弱），后期逐渐增至 1（正则化强），避免训练初期正则化对模型的干扰。
    '''
    annealing_coef = min(1, global_step / annealing_step)
    '''
    1 - label 是独热标签的 “反向”（真实类别位置为 0，其他为 1）。
    E * (1 - label) 对 “非真实类别” 的证据进行调整，再+1 得到新的证据张量 alp，用于后续的 KL 散度计算。
    '''
    alp = E * (1 - label) + 1
    #KL(alp, c) 计算 alp 与 “均匀狄利克雷分布” 的KL 散度（衡量 alp 与 “无偏分布” 的差异）。
    B = annealing_coef * KL(alp, c)

    return (A + B)


#用于分类任务中的分布鲁棒性优化，结合了证据损失和 KL 散度损失，核心逻辑是区分 “分布内样本” 和 “分布外（OOD）样本”，对 OOD 样本施加正则化以提升模型泛化能力。
def regularization(input, target, current_epoch, args):

    batch_num, class_num = input.shape
    logits = input
    '''
    ood_num：根据超参数 args.eta（OOD 样本比例）计算当前批次中需处理的 OOD 样本数量（eta×批大小）
    '''
    ood_num = int(args.eta * batch_num)
    if (ood_num == 0 or current_epoch < args.ours_start_step):
        return torch.mean(evidence_loss(target, input, args.class_num, current_epoch, args.lambda_epochs))
     #若 OOD 样本数为 0，或当前训练轮次 current_epoch 小于预设的 “正则化开始轮次” args.ours_start_step，则直接返回证据损失的平均值（此时不进行 OOD 正则化，只做基础分类损失优化）
    ce_loss = evidence_loss(target, input, args.class_num, current_epoch, args.lambda_epochs)#用于衡量模型对 “分布内样本” 的分类准确性和分布鲁棒性

    # order according to ce loss
    #损失大的样本更可能是 “分布外（OOD）样本”（模型对其分类信心低、分布异常），因此排在前 ood_num 位的样本被视为 OOD 样本。
    rk = torch.argsort((ce_loss).squeeze(-1), descending=True)#压缩维度后，对每个样本的证据损失进行降序排序，得到排序索引 rk
    # ce_loss = torch.sum(ce_loss[rk[ood_num:]])

    # second term: kl divergence loss
    log_probs = logits[rk[:ood_num]]
    #构造均匀分布的目标张量（每个类别概率为 1/类别数），代表 “无偏的类别分布”。
    target_distribution = torch.tensor([[1.0 / class_num] * class_num]).expand_as(log_probs).cuda()
    #计算 OOD 样本的 logits 与均匀分布的KL 散度，衡量 OOD 样本的分布与 “无偏分布” 的差异。reduction='sum' 表示对所有 OOD 样本的 KL 散度求和。
    kl_loss = F.kl_div(log_probs, target_distribution, reduction='sum')

    # loss = torch.mean(ce_loss + args.beta * kl_loss)
    # w = torch.exp(-u)
    loss = torch.mean(ce_loss)


    return loss
