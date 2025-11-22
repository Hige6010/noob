# Author: Xiaoli Wang
# Email: xiaoliw1995@gmail.com
# @Time 2023/11/22
import os

import scipy.io as scio
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from get_sn import *
from torch.utils.data import Dataset, Sampler
from scipy.spatial.distance import cdist

reg_param  = 1e-3

cuda = True if torch.cuda.is_available() else False

def get_samples(x, y, sn, train_index, test_index, use_mean=False):
    """
    Retrieve the set of the k nearest samples with missing data on the training dataset.
    :param x: dataset: view_num * (dataset_num, dim,)
    :param y: label: (dataset_num,)
    :param sn: missing index matrix: (dataset_num, view_num,)
    :param train_index: (train_num,)
    :param test_index: (test_num,)
    :return:
    """
    view_num = len(x)
    train_num, test_num = train_index.shape[0], test_index.shape[0]
    print(train_num, test_num)

    print("Fill the missing views in the training set")
    # step1: obtain the training set
    # 从视图缺失掩码矩阵中提取训练集样本的缺失情况
    sn_train = sn[train_index]
    #从原始视图数据中筛选出训练集对应的样本数据
    x_train = [x[v][train_index] for v in range(view_num)]
    #从原始标签数组中筛选出对应的训练集样本的标签
    y_train = y[train_index]

    MeanView = []
    for _ in range(view_num):
        #筛选第_个视图中“未缺失”的样本特征
        valid_view = x_train[_][sn_train[:, _].astype(bool)]
        #计算该视图有效样本的特征均值
        mean = np.mean(valid_view, axis=0)
        if use_mean == True:
            print("use mean")
            #若启用均值填充，将该视图的均值加入填充列表
            MeanView.append(mean)
        else:
            # 若不启用，用零向量填充
            MeanView.append(np.zeros_like(mean))

    # step4: fill incomplete samples
    #np.where(condition) 会返回满足条件的元素的索引。在这里，它会返回所有 “存在视图缺失的训练集样本” 的索引，最终赋值给 x_train_miss_index。
    # 筛选“存在视图缺失”的训练集样本索（沿着这个维度求和）
    x_train_miss_index = np.where(np.sum(sn_train, axis=1) < view_num)[0]

    for i in x_train_miss_index:
        #筛选第i个样本中“缺失的视图”索引
        miss_view_index = np.nonzero(sn_train[i] == 0)[0]
        for v in miss_view_index:   # 每个样本中缺失的视图索引
            x_train[v][i] = MeanView[v]

    # x_train = process_data(x_train, view_num)
    print("Fill the missing views in the test set")
    sn_test = sn[test_index]
    x_test = [x[_][test_index] for _ in range(view_num)]
    y_test = y[test_index]

    # calculate mean value of each view without missing view
    # MeanTest = [np.sum(np.expand_dims(sn_test[:, _], axis=-1) * x_test[_]) / np.sum(sn_test[_]) for _ in range(view_num)]
    MeanTest = []
    for _ in range(view_num):
        valid_view = x_test[_][sn_test[:, _].astype(bool)]
        mean = np.mean(valid_view, axis=0)
        if use_mean == True:
            MeanTest.append(mean)
        else:
            MeanTest.append(np.zeros_like(mean))
    # step4: fill incomplete samples
    x_test_miss_index = np.where(np.sum(sn_test, axis=1) < view_num)[0]

    for i in x_test_miss_index:
        test_miss_view_index = np.nonzero(sn_test[i] == 0)[0]
        for v in test_miss_view_index:  # 每个样本中缺失的视图索引
            x_test[v][i] = MeanTest[v]

    # x_test = process_data(x_test, view_num)
    x_new = [np.concatenate((x_train[i], x_test[i]), axis=0) for i in range(view_num)]
    x_train = [x_new[_][:train_num] for _ in range(view_num)]
    x_test = [x_new[_][train_num:] for _ in range(view_num)]

    return x_train, y_train, x_test, y_test, sn_train, sn_test

#按照指定比例从原始数据中随机拆分出训练样本和测试样本的索引，保证数据划分的随机性和可复现性
def split_dataset(Y, p, seed=999):
    '''
    Split train and test dataset
    :param seed: Random seed
    :param p: proportion of samples for training
    :param Y: the original class indexes
    :return: partition: include train_idx and test_idx
    '''
    # 设置NumPy的随机种子，保证划分结果可复现
    np.random.seed(seed=seed)
    # 压缩Y的维度（若Y是二维数组，压缩为一维）
    Y = np.squeeze(Y)
    # 生成样本的索引数组（0到len(Y)-1）
    Y_idx = np.array([x for x in range(len(Y))])
    #np.ceil是NumPy的“向上取整” 函数，确保计算结果是大于等于该值的最小整数
    num_train = np.int_(np.ceil(len(Y_idx) * p))# 从所有样本索引中随机选择num_train个，不重复
    #replace=false不允许重复选择
    train_idx_idx = np.random.choice(len(Y_idx), num_train, replace=False)# 所有样本索引中随机选择指定数量的样本，作为训练集的索引
    # train_idx_idx = np.arange(num_train)
    train_idx = Y_idx[train_idx_idx]
    test_idx = np.array(list(set(Y_idx.tolist()) - set(train_idx.tolist())))# 利用集合的 “差集” 操作（-），得到所有属于总样本但不属于训练集的样本索引
    partition = {'tr': train_idx, 'te': test_idx}# 封装训练集和测试集索引为字典
    return partition

def process_data(X, n_view, if_meanMax=False):

    #归一化
    if if_meanMax == True:
        if (n_view == 1):
            m = np.mean(X)
            mx = np.max(X)
            mn = np.min(X)
            X = (X - m) / (mx - mn)
        else:
            for i in range(n_view):
                m = np.mean(X[i])
                mx = np.max(X[i])
                mn = np.min(X[i])
                X[i] = (X[i] - m) / (mx - mn)
    else:
        X = [StandardScaler().fit_transform(X[i]) for i in range(n_view)]]#标准标准化，实现每个视图内部特征的尺度统一，为后续多视图融合或模型训练奠定基础。
    return X


#加载并预处理多视图数据集，包括训练集、测试集的特征和标签
def preprocess_data(data_folder, view_list):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_te.csv"), delimiter=','))
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        # 将训练集和测试集的特征在样本维度（axis=0）拼接，形成完整数据集
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    # data_mat_list = process_data(data_mat_list, n_view=num_view)
    # data_tensor_list = []
    # for i in range(len(data_mat_list)):
    #     data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
    #     if cuda:
    #         data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = np.array(list(range(num_tr)))
    idx_dict["te"] = np.array(list(range(num_tr, (num_tr + num_te))))
    # data_train_list = []
    # data_test_list = []
    # data_all_list = []
    # for i in range(len(data_tensor_list)):
    #     data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
    #     data_test_list.append(data_tensor_list[i][idx_dict["te"]].clone())
    #     data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
    #                                     data_tensor_list[i][idx_dict["te"]].clone()), 0))
    labels = np.concatenate((labels_tr, labels_te))

    return data_mat_list, labels, idx_dict

def load_data(path, name, miss_rate):
    """
    :param path: dataset path
    :param name: dataset name
    :return:
    X: python list containing all views, where each view is represented as numpy array
    Y: ground_truth labels represented as numpy array
    dims: python list containing dimensions of each view
    n_views: number of views
    n_samples: number of samples
    class_num: number of category
    """
    filepath = path + name + '.mat'
    print(filepath)
    f = scio.loadmat(filepath)# 加载.mat文件（使用scipy.io的loadmat函数）

    gt = (f['gt']).astype(np.int32)# 提取真实标签并转换为int32类型 标签是离散的类别索引（如 0、1、2 代表不同类别），不需要小数精度，使用整数类型（int32）可以节省存储开销，同时避免浮点数可能引入的精度问题。

    if gt.min() == 1:
        gt = gt - 1
    else:
        gt = gt
    class_num = gt.max() + 1

    data = (f['X'])  # dim * num
    X = []

    for x in data[0]:
        X.append(x.astype(np.float64))

    n_sample = len(X[0][0])
    n_view = len(X)

    dims = []
    for i in range(n_view):
        X[i] = X[i].T
        dims.append(X[i].shape[1])

    Sn = get_sn(n_view, n_sample, miss_rate).astype(np.float32)

    return X, gt, Sn, dims, n_view, n_sample, class_num

def get_data(path, name, miss_rate, use_mean=True):

    data_list, Y, Sn, dims, n_view, data_size, class_num = load_data(path, name, miss_rate)
    # step 2. split train/test dataset and dataloader
    X = process_data(data_list, n_view, if_meanMax=False)  # StandardScaler
    idx_dict = split_dataset(Y, p=0.8, seed=999)  # dict{'train', 'test'}

    X_train, Y_train, X_test, Y_test, Sn_train, Sn_test = get_samples(x=X, y=Y, sn=Sn,
                                                                      train_index=idx_dict['tr'],
                                                                      test_index=idx_dict['te'],
                                                                      use_mean=use_mean
                                                                      )

    return X_train, Y_train, X_test, Y_test, Sn_train, Sn_test, dims, class_num

if __name__ == '__main__':

    data_path = "./dataset/"
    name = 'ROSMAP'
    miss_rate=0.2

    data_list, Y, Sn, dims, n_view, data_size, class_num = load_data(data_path, name, miss_rate=miss_rate)
    X = process_data(data_list, n_view, if_meanMax=False)  # StandardScaler
    idx_dict = split_dataset(Y, p=0.8, seed=999)  # dict{'train', 'test'}

    X_train, Y_train, X_test, Y_test, Sn_train, Sn_test = get_samples(x=X, y=Y, sn=Sn,
                                                                      train_index=idx_dict['tr'],
                                                                      test_index=idx_dict['te'],
                                                                      use_mean=False

                                                                      )
