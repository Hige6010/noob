import torch
from torch.optim import Adam
from dataloader import get_data, get_sn
from utils import *
from EarlyStopping_hand import EarlyStopping
from Models.Teacher import T_model
from Models.Student import S_model
from Models.IMv_teacher import teacher
from Models.IMv_kd import kd


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
                              shuffle=True,
                              drop_last=True,
                              collate_fn=partial_mv_tabular_collate)

    test_loader = DataLoader(dataset=partial_mv_dataset(X_test, Sn_test, Y_test), batch_size=args.batch_size,
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
