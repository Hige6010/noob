# Author: Xiaoli Wang
# Email: xiaoliw1995@gmail.com
# @Time 2024/4/21
# !/user/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import os

import sklearn.metrics
import torch.optim
import torch.nn.functional as F
import random
import numpy
# import hiddenlayer as hl
import time

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, precision_recall_fscore_support
# import wandb
from torch import optim

from utils import AverageMeter, late_fusion
from loss import MAD, evidence_loss, regularization
from utils import data_write_csv
from EarlyStopping_hand import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

# wandb.init(project="my-project", entity="wangxiaoli")
# wandb.init(project="my-project", name="wandb-demo")

    # 使用wandb.log 记录你想记录的指标
    # wandb.log({
    #     "Test Accuracy": acc * 100.
    # })

# wandb.watch_called = False

def test(StudentModel, test_loader, args, file_path):
    # print("----------KD StudentModel Test------------")
    #从训练模式（Training Mode） 切换到评估模式（Evaluation Mode），以确保在测试 / 推理时得到正确的结果。
    StudentModel.eval()
    # kd_path = f'SaveModel/{args.data_name}/save_kd_{args.miss_rate}.pt'
    # if os.path.exists(kd_path):
    #     StudentModel.load_state_dict(torch.load(kd_path))

    data_num, correct_num = 0, 0
    # 真实标签、预测概率、预测类别
    target_set, logits_set, lbs_set = [], [], []
    for batch_idx, (data, sn, target) in enumerate(test_loader):
        for v_num in range(len(data)):
            # 将多视图数据转换为CUDA张量（GPU加速）
            data[v_num] = data[v_num].float().cuda()
        target = target.long().cuda()
        sn = sn.float().cuda()
        data_num += target.size(0)
        ## 关闭梯度计算（测试阶段无需反向传播）
        with torch.no_grad():
            output, fm = StudentModel(data, sn)
            if isinstance(output, list):
                for v in range(len(data)):
                    # 对每个视图的输出应用掩码（过滤缺失视图）
                    output[v] = output[v] * sn[:, v].unsqueeze(-1)
                # 融合多视图结果：求和后计算softmax得到最终概率
                logits = torch.softmax(torch.sum(torch.stack(output, dim=0), dim=0), dim=1)
            else:
                logits = torch.softmax(output, dim=1)
            # 从概率中获取预测类别（取最大值对应的索引）
            #torch.max(logits, dim=1) 沿着类别维度计算每个样本的最大概率及其索引
            _, lbs = torch.max(logits, dim=1)
            correct_num = correct_num + (lbs == target).sum().item()
            target_set.append(target)
            logits_set.append(logits)
            lbs_set.append(lbs)
    # 合并所有批次的结果（从列表转为张量）
    target_all = torch.concat(target_set, dim=0)
    logits_all = torch.concat(logits_set, dim=0)
    lbs_all = torch.concat(lbs_set, dim=0)
    # 计算多类别评估指标
    # 宏平均：计算每个类别的指标后取平均
    #用于一次性计算多个分类评估指标，返回一个包含 4 个元素的元组：
    '''
    准确率（Accuracy）：正确预测数 / 总样本数
    精确率（Precision）：预测为正的样本中实际为正的比例
    召回率（Recall）：实际为正的样本中被正确预测的比例
     F1 分数：精确率和召回率的调和平均，平衡两者
    '''
    precision, recall, f1_score, _ = precision_recall_fscore_support(target_all.cpu().numpy(), lbs_all.cpu().numpy(), average='macro')
    # Print the results for each class
    # for i, (p, r, f1) in enumerate(zip(precision, recall, f1_score)):
    #     print(f"Class {i}: Precision = {p:.2f}, Recall = {r:.2f}, F1 Score = {f1:.2f}")
    acc = correct_num / data_num
    # acc = accuracy_score(target_all.cpu().numpy(), lbs_all.cpu().numpy())
    #计算AUC（多类别采用一对一策略）
    #AUC：ROC 曲线下面积，衡量模型区分不同类别的能力
    auc = roc_auc_score(target_all.cpu().numpy(), logits_all.cpu().numpy(), multi_class='ovo')
    print(f"=====================Student test acc:{acc}, test f1:{f1_score}, test precision:{precision}, test recall:{recall}, test auc:{auc}")
    # data_write_csv(file_path, 'with_RL: ' + str(args.use_rl_eta) + ' miss_rate: '
    #                + str(args.miss_rate) + ' eta: ' + str(args.eta) + ' ood: ' + str(args.ood)
    #                + ' test--ACC: ' + str(f"{acc:.4f}"))
    return acc, f1_score

#用于创建学习率调度器
def get_scheduler(optimizer, args):
    #ReduceLROnPlateau 类型：这是 PyTorch 中的一种学习率调度策略，当某个指标（如验证准确率）停止改善时，自动降低学习率。
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )

def kd(device, file_path, args, TeacherModel, StudentModel, train_loader, test_loader):

    print("-------------KDModel start---------------")
    # 蒸馏损失计算函数,用于计算教师模型和学生模型之间的 “知识蒸馏损失”（衡量两者输出 / 特征的差异）
    kd_criterion = MAD()
    optimizer = torch.optim.Adam(StudentModel.parameters(), lr=args.kd_lr, weight_decay=1e-5)
    scheduler = get_scheduler(optimizer, args)
    # 可视化：记录训练过程的指标
    # wandb.watch(StudentModel, log="all")

    best_test_acc, best_test_f1, best_epoch = 0., 0., 0
    for epoch in range(args.kd_epochs):
        StudentModel.train()
        loss_meter = AverageMeter()
        ce_meter = AverageMeter()
        acc_meter = AverageMeter()
        kd_meter = AverageMeter()
        data_num, correct_num = 0, 0

        for batch_idx, (data, sn, target) in enumerate(train_loader):
            # refresh the optimizer
            optimizer.zero_grad()

            data_t = []
            data_s = []
            for v_num in range(len(data)):
                data[v_num] = data[v_num].float().cuda()
                data_t.append(data[v_num].clone())
                data_s.append(data[v_num].clone())

            data_num = target.size(0)
            gt = target.clone()
            target = target.long().cuda()# 学生模型用的类别标签

            # NOTE: 把gt改成one_hot
            gt_onehot = F.one_hot(gt.to(torch.int64), args.class_num).float().cuda()# 教师模型用的one-hot标签
            sn = sn.float().cuda()# 视图缺失掩码

            # 取出teacher model的EncX 和 predict
            with torch.no_grad():
                logit_t, fm_t, _ = TeacherModel(data_t, gt=gt_onehot, src_mask=sn)

            # sn_fix 是为了增加缺失而设置的一个固定的值
            output, fm_s = StudentModel(data_s, src_mask=sn)  # logits is a list that without softmax
            # kd_loss = kd_criterion.new_kd(fm_t, fm_s, logit_t, output)
            # NOTE: 注意使用的蒸馏损失是否加权了
            # kd_loss = kd_criterion.forward(fm_t, fm_s, logit_t)
            # 计算蒸馏损失（学生特征与教师特征的差异）
            kd_loss = kd_criterion.mse_loss(fm_t, fm_s)

            _, lbs = torch.max(F.log_softmax(output, dim=-1), dim=1)

            data_dim = target.size(0)
            # ce_loss = late_fusion(output, target, sn, epoch, args)[0]

            # ce_loss = torch.mean(evidence_loss(target, output, args.class_num, epoch, args.lambda_epochs))
            # ce_loss = regularization(output, target, epoch, args)
            # 计算分类损失（学生预测与真实标签的差异）
            ce_loss = F.cross_entropy(output, target, reduction='mean')

            correct_num = (lbs == target).sum().item()
            acc_meter.update(correct_num / target.size(0))
            # 总损失 = 分类损失 + 蒸馏损失（带权重）
            loss = ce_loss + args.lam * kd_loss
            # loss = ce_loss #+ args.lam * kd_loss

            # compute gradients and take step
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            ce_meter.update(ce_loss.item())
            kd_meter.update(kd_loss.item())

        # acc = correct_num / data_num
        print(f"{epoch}==>kd_train_acc:{acc_meter.avg} ce_loss:{ce_meter.avg:4f} kd_loss:{kd_meter.avg:4f} loss:{loss_meter.avg:4f}")

        # if acc_meter.avg > baseline:
        #     baseline = acc_meter.avg
        #     if not os.path.exists(f'./SaveModel/{args.data_name}'):
        #         os.mkdir(f'./SaveModel/{args.data_name}')
        #     path = f'./SaveModel/{args.data_name}/save_kd_{args.miss_rate}' + '.pt'
        #     torch.save(StudentModel.state_dict(), path)

        # early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        # early_stopping(loss_meter.avg * (-1), StudentModel, args.data_name, args.miss_rate, Name='SNet')
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

        # wandb.log({
        #     "Train loss": loss_meter.avg,
        #     "ce_loss": ce_meter.avg,
        #     "kd_loss": kd_meter.avg
        # })
        # test

        test_acc, test_f1 = test(StudentModel, test_loader, args, file_path)
        # scheduler.step(test_acc)

        if best_test_acc < test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            if not os.path.exists(f'./SaveModel/{args.data_name}'):
                os.mkdir(f'./SaveModel/{args.data_name}')
            path = f'./SaveModel/{args.data_name}/save_student_{args.miss_rate}' + '.pt'
            torch.save(StudentModel.state_dict(), path)
        if best_test_f1 < test_f1:
            best_test_f1 = test_f1

    # if not os.path.exists(f'./SaveModel/{args.data_name}'):
    #     os.mkdir(f'./SaveModel/{args.data_name}')
    # path = f'./SaveModel/{args.data_name}/save_vanilla_{args.miss_rate}' + '.pt'
    # torch.save(StudentModel.state_dict(), path)

    return best_test_acc, best_test_f1, best_epoch





