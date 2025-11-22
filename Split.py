import os
import torch
import argparse
from runner import run
from utils import data_write_csv, setup_seed
from dataloader import get_data

dataPath = "../IMv_Project"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train')

    parser.add_argument('--path', default=os.path.join(dataPath, '/home/wangxiaoli/xiaoli/IMv_project/dataset/'), type=str)
    # parser.add_argument('--path', type=str, default=os.path.join(dataPath, 'C:/Users/xiaol/PycharmProjects/IMv_Project/dataset/'))
    # Training info
    parser.add_argument('--data_name', type=list, default=['Reuter', 'GRZA02', 'LandUse_21'])
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                            help='learning rate [default: 1e-3]')
    parser.add_argument('--miss_rate', default=0., type=float)
    parser.add_argument('--patience', type=int, default=10, metavar='LR',
                        help='parameter of Earlystopping [default: 30]')
    parser.add_argument("--teacher_epochs", default=100)
    parser.add_argument('--kd_lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate [default: 1e-3]')
    parser.add_argument("--kd_epochs", default=100, type=int)
    parser.add_argument("--lr_factor", type=float, default=0.9)
    parser.add_argument("--lr_patience", type=int, default=5)

    # Model hyperparameters
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--label_embedd", type=int, default=512)

    # Algorithm hyperparameters
    parser.add_argument('--eta', default=0.05, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--lambda_epochs', default=50, type=int)
    parser.add_argument('--ours_start_step', default=0, type=int)
    parser.add_argument('--lam', default=10, type=float, help="balance factor of kd_loss")
    parser.add_argument('--use_rl_eta', action='store_true', default=False)
    parser.add_argument('--ood', default=False, type=bool)  # If use ood strategy

    # args, unknown_args = parser.parse_known_args()
    args, unknown_args = parser.parse_known_args()


    # NOTE：Make uniform dataset for all compared datasets
    # Make uniform dataset
    setup_seed(42)
    miss_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    data_name = ['Caltech101-20']
    for dn in data_name:
        args.data_name = dn
        for mr in miss_rate:
            args.miss_rate = mr
            X_train, Y_train, X_test, Y_test, Sn_train, Sn_test, dims, class_num = get_data(
                args.path,
                args.data_name,
                args.miss_rate,
                use_mean=True)
    #
    #         # NOTE: 服务器路径
            if not os.path.exists(f'/home/wangxiaoli/xiaoli/IMv_project/MyDataset/{args.data_name}'):
                os.mkdir(f'/home/wangxiaoli/xiaoli/IMv_project/MyDataset/{args.data_name}')
            torch.save(X_train, f'/home/wangxiaoli/xiaoli/IMv_project/MyDataset/{args.data_name}/X_train_{args.miss_rate}.pt')
            torch.save(X_test, f'/home/wangxiaoli/xiaoli/IMv_project/MyDataset/{args.data_name}/X_test_{args.miss_rate}.pt')
            torch.save(Y_train, f'/home/wangxiaoli/xiaoli/IMv_project/MyDataset/{args.data_name}/Y_train_{args.miss_rate}.pt')
            torch.save(Y_test, f'/home/wangxiaoli/xiaoli/IMv_project/MyDataset/{args.data_name}/Y_test_{args.miss_rate}.pt')
            torch.save(Sn_train, f'/home/wangxiaoli/xiaoli/IMv_project/MyDataset/{args.data_name}/Sn_train_{args.miss_rate}.pt')
            torch.save(Sn_test, f'/home/wangxiaoli/xiaoli/IMv_project/MyDataset/{args.data_name}/Sn_test_{args.miss_rate}.pt')

            # # NOTE: for other methods
            if not os.path.exists(f'./MyDataset/{args.data_name}'):
                os.mkdir(f'./MyDataset/{args.data_name}')
            torch.save(X_train, f'./MyDataset/{args.data_name}/X_train_{args.miss_rate}.pt')
            torch.save(X_test, f'./MyDataset/{args.data_name}/X_test_{args.miss_rate}.pt')
            torch.save(Y_train, f'./MyDataset/{args.data_name}/Y_train_{args.miss_rate}.pt')
            torch.save(Y_test, f'./MyDataset/{args.data_name}/Y_test_{args.miss_rate}.pt')
            torch.save(Sn_train, f'./MyDataset/{args.data_name}/Sn_train_{args.miss_rate}.pt')
            torch.save(Sn_test, f'./MyDataset/{args.data_name}/Sn_test_{args.miss_rate}.pt')

            # # NOTE: for CPM, Do not forget modify dataloader.process_data
            # if not os.path.exists(f'./MyDataset/CPM/{args.data_name}'):
            #     os.mkdir(f'./MyDataset/CPM/{args.data_name}')
            # torch.save(X_train, f'./MyDataset/CPM/{args.data_name}/X_train_{args.miss_rate}.pt')
            # torch.save(X_test, f'./MyDataset/CPM/{args.data_name}/X_test_{args.miss_rate}.pt')
            # torch.save(Y_train, f'./MyDataset/CPM/{args.data_name}/Y_train_{args.miss_rate}.pt')
            # torch.save(Y_test, f'./MyDataset/CPM/{args.data_name}/Y_test_{args.miss_rate}.pt')
            # torch.save(Sn_train, f'./MyDataset/CPM/{args.data_name}/Sn_train_{args.miss_rate}.pt')
            # torch.save(Sn_test, f'./MyDataset/CPM/{args.data_name}/Sn_test_{args.miss_rate}.pt')
