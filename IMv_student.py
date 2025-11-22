# Author: Xiaoli Wang
# Email: xiaoliw1995@gmail.com
# @Time 2024/4/21
# !/user/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from utils import AverageMeter, late_fusion


def student(device, args, StudentModel, train_loader, test_loader, Sn_loader, Sn_test_loader):

    print("---------StudentModel start----------")
    model = StudentModel
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    loss_meter = AverageMeter()
    # acc_meter = AverageMeter()
    data_num, correct_num = 0, 0

    print("----------StudentModel Train------------")
    for epoch in range(args.student_epochs):
        model.train()
        for batch_idx, ((data, sn, target), sn_fix) in enumerate(zip(train_loader, Sn_loader)):

            for v_num in range(len(data)):
                data[v_num] = data[v_num].float().cuda()
            data_num += target.size(0)
            gt = target.clone()
            target = target.long().cuda()
            # NOTE: 把gt改成one_hot
            gt_onehot = F.one_hot(gt.to(torch.int64), args.class_num).float().cuda()
            sn = sn.long().cuda()
            sn_fix = sn_fix.long().to(device)
            # refresh the optimizer
            optimizer.zero_grad()
            # NOTE: sn_fix 是为了增加缺失而设置的一个固定的值
            output, _ = model(data, sn_fix, src_mask=sn, feature_fusion=args.feature_fusion,
                           gt=None)  # logits is a list that without softmax
            # NOTE: 修改为决策层融合，注意mask缺失视图的值
            _, lbs = torch.max(output, dim=1)
            loss = late_fusion(output, target, sn, epoch, args)[0]
            correct_num += (lbs == target).sum().item()

            # compute gradients and take step
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())

        acc = correct_num / data_num
        print(f"{epoch}==>train_acc:{acc}")

    print("----------StudentModel Test------------")
    model.eval()
    data_num, correct_num = 0, 0
    for batch_idx, ((data, sn, target), sn_fix_test) in enumerate(zip(test_loader, Sn_test_loader)):
        for v_num in range(len(data)):
            data[v_num] = data[v_num].float().cuda()
        target = target.long().cuda()
        sn = sn.long().cuda()
        sn_fix_test = sn_fix_test.long().cuda()
        data_num += target.size(0)
        with torch.no_grad():
            output, _ = model(data, sn_fix_test, sn, args.feature_fusion)
            # if isinstance(output, list):
            #     for v in range(len(data)):
            #         output[v] = output[v] * sn[:, v].unsqueeze(-1)
            #     logits = torch.log_softmax(torch.sum(torch.stack(output, dim=0), dim=0), dim=1)
            # else:
            #     logits = torch.log_softmax(output, dim=1)

            _, lbs = torch.max(output, dim=1)
            correct_num = correct_num + (lbs == target).sum().item()
    acc = correct_num / data_num
    print(f"Student test acc:{acc}")