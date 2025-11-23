import torch
from torch.optim import Adam
from dataloader import get_data, get_sn
from utils import *
from EarlyStopping_hand import EarlyStopping
from Models.Teacher import T_model
from Models.Student import S_model
from Models.IMv_teacher import teacher
from Models.IMv_kd import kd

'''
批次大小一致：
在训练过程中，模型的权重更新是基于每一个批次（batch）的数据计算的梯度。大多数神经网络层（如全连接层、卷积层）和优化器都假设每个批次的输入数据具有相同的形状（shape）。如果最后一个批次的样本数量与其他批次不同，可能会导致：
计算错误：某些操作，特别是涉及到矩阵运算或自定义损失函数时，可能会因为输入维度不匹配而报错。
梯度不稳定：即使程序不报错，一个更小的批次计算出的梯度可能会与标准批次大小的梯度有较大差异，这会引入训练噪声，导致模型收敛路径不稳定，可能需要更长的训练时间才能达到理想效果。
效率考虑：
训练是一个迭代过程，通常会进行成千上万次。处理一个只包含一两个样本的微型批次，其带来的模型提升微乎其微，但仍然需要花费一次完整的前向和反向传播的计算资源。在这种情况下，为了效率，丢弃它是明智的。
'''

def run(args, unknown_args, device, file_path):
    X_train = torch.load(f'./MyDataset/{args.data_name}/X_train_{args.miss_rate}.pt')
    X_test = torch.load(f'./MyDataset/{args.data_name}/X_test_{args.miss_rate}.pt')
    Y_train = torch.load(f'./MyDataset/{args.data_name}/Y_train_{args.miss_rate}.pt')
    Y_test = torch.load(f'./MyDataset/{args.data_name}/Y_test_{args.miss_rate}.pt')
    Sn_train = torch.load(f'./MyDataset/{args.data_name}/Sn_train_{args.miss_rate}.pt')
    Sn_test = torch.load(f'./MyDataset/{args.data_name}/Sn_test_{args.miss_rate}.pt')

    args.dims = [X_train[v].shape[1] for v in range(len(X_train))]
    args.class_num = np.max(Y_train) + 1

    train_loader = DataLoader(dataset=partial_mv_dataset(X_train, Sn_train, Y_train),
                              batch_size=args.batch_size,
                              shuffle=True,#训练集打乱数据，增强泛化
                              shuffle=True,
                              drop_last=True,
                              collate_fn=partial_mv_tabular_collate) #是自定义数据拼接函数，适配多视图 / 缺失数据的批量处理。
                              collate_fn=partial_mv_tabular_collate)

    test_loader = DataLoader(dataset=partial_mv_dataset(X_test, Sn_test, Y_test), batch_size=args.batch_size,
                             shuffle=False,#测试集保持顺序，便于结果对齐
                             drop_last=False,#drop_last 是一个布尔参数，用于控制当数据集的样本数不能被 batch_size 整除时，是否丢弃最后一个不完整的批次，此处确保所有测试样本都被评估，得到最准确的测试结果。
                             shuffle=False,
                             drop_last=False,
                             collate_fn=partial_mv_tabular_collate)

    TeacherModel = T_model(args.dims, label_embedd=args.label_embedd, d_model=args.d_model,
                           n_layers=args.n_layers, heads=4,
                           classes_num=args.class_num, tau=args.tau,
                           dropout=0.)
    TeacherModel = TeacherModel.to(device)

    StudentModel = S_model(args.dims, d_model=args.d_model,
                           n_layers=args.n_layers, heads=4,
                           classes_num=args.class_num, tau=args.tau,
                           dropout=0.)
    StudentModel = StudentModel.to(device)

    teacher_path = f'C:/Users/xiaol/PycharmProjects/IMv_Project/SaveModel/{args.data_name}/save_TNet_{args.miss_rate}.pt'
    # if os.path.exists(teacher_path):
    #     print("load teacher model")
    #     TeacherModel.load_state_dict(torch.load(teacher_path), False)
    #     best_test_acc, best_test_f1, best_epoch = kd(device, file_path, args, TeacherModel, StudentModel, train_loader,
    #                                                  test_loader)
    #
    # else:
    # 从头开始训练TeacherModel
    teacher(device, args, TeacherModel, train_loader)
    best_test_acc, best_test_f1, best_epoch = kd(device, file_path, args, TeacherModel, StudentModel, train_loader,
                                                 test_loader)
    #
    return best_test_acc, best_test_f1, best_epoch
