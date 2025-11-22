# Author: Xiaoli Wang
# Email: xiaoliw1995@gmail.com
# @Time 2024/4/21
# !/user/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import os

import torch.optim
import torch.nn.functional as F

import utils
from EarlyStopping_hand import EarlyStopping
from utils import late_fusion, AverageMeter
from loss import regularization


def teacher(device, args, TeacherModel, train_loader):

    print("---------TeacherModel start----------")
    optimizer = torch.optim.Adam(TeacherModel.parameters(), lr=args.lr, weight_decay=1e-5)

    loss_meter = AverageMeter()
    # acc_meter = AverageMeter()
    data_num, correct_num, baseline = 0, 0, 0.

    gtEmbedding, MultiviewEmbedd, gt_list = [], [], []
    for epoch in range(args.teacher_epochs):
        TeacherModel.train()
        for batch_idx, (data, sn, target) in enumerate(train_loader):
            for v_num in range(len(data)):
                data[v_num] = data[v_num].float().cuda()
            data_num += target.size(0)
            gt = target.clone()
            target = target.long().cuda()
            # NOTE: 把gt改成one_hot
            gt_onehot = F.one_hot(gt.to(torch.int64), args.class_num).float().cuda()
            sn = sn.float().cuda()
            # refresh the optimizer
            optimizer.zero_grad()
            # NOTE: sn_fix 是为了增加缺失而设置的一个固定的值
            output, EncX, _ = TeacherModel(data, gt=gt_onehot, src_mask=sn, save=False)    # logits is a list that without softmax
            # NOTE: 修改为决策层融合，注意mask缺失视图的值
            _, lbs = torch.max(F.log_softmax(output, dim=1), dim=1)
            # loss = late_fusion(output, target, sn, epoch, args)[0]
            # loss = regularization(output, target, epoch, args)
            loss = F.cross_entropy(output, target)
            correct_num += (lbs == target).sum().item()

            # compute gradients and take step
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())

        acc = correct_num / data_num
        print(f"{epoch}==>train_acc:{acc}")

    # labelEmbedd = torch.cat(gtEmbedding, dim=0)
    # ViewEmbedd = torch.cat(MultiviewEmbedd, dim=0)
    # gt_new = torch.cat(gt_list, dim=0)
    # torch.save(labelEmbedd, 'labelEmbedd.pt')
    # torch.save(ViewEmbedd, 'ViewEmbedd.pt')
    # torch.save(gt_new, 'label.pt')


        # early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        # early_stopping(loss_meter.avg * (-1), TeacherModel, args.data_name, args.miss_rate, Name='TNet')
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     TeacherModel.eval()
        #     break

        if acc > baseline:
            baseline = acc
            if not os.path.exists(f'./SaveModel/{args.data_name}'):
                os.mkdir(f'./SaveModel/{args.data_name}')
            path = f'./SaveModel/{args.data_name}/save_teacher_{args.miss_rate}' + '.pt'
            torch.save(TeacherModel.state_dict(), path)

    TeacherModel.eval()


