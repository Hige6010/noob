import torch
import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0# 计数器，用于记录验证损失连续没有改善的 epoch 数。
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf#历史最小验证损失
        self.delta = delta

    def __call__(self, val_loss, model, data_name, miss_rate, Name='TNet'):

        score = val_loss#将验证损失作为评估指标。

        #if验证第一次调用
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, data_name, miss_rate, Name='TNet')
        #如果成立，说明这一轮损失没有得到改善
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, data_name, miss_rate, Name='TNet')
            self.counter = 0

    def save_checkpoint(self, val_loss, model, data_name, miss_rate, Name='TNet'):
        '''
        Saves model when validation loss decrease.
        '''
        if self.verbose:
            pass
            # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), f'/home/wangxiaoli/xiaoli/IMv_project/SaveModel/{data_name}/save_{Name}_{miss_rate}.pt')     # 这里会存储迄今最优模型的参数
        #使用 PyTorch 的 torch.save 函数将模型的状态字典（state_dict）保存到 self.path 指定的文件中。state_dict 包含了模型所有层的权重和偏置。
        self.val_loss_min = val_loss
