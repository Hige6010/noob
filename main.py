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

    # parser.add_argument('--path', default=os.path.join(dataPath, '/home/wangxiaoli/xiaoli/IMv_project/dataset/'), type=str)
    parser.add_argument('--path', type=str, default=os.path.join(dataPath, 'C:/Users/xiaol/PycharmProjects/IMv_Project/dataset/'))
    # Training info
    parser.add_argument('--data_name', type=str, default='Scene15')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                            help='learning rate [default: 1e-3]')
    parser.add_argument('--miss_rate', default=0., type=float)
    parser.add_argument('--patience', type=int, default=5, metavar='LR',
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
    parser.add_argument("--n_layers", type=int, default=1)

    # Algorithm hyperparameters
    parser.add_argument('--eta', default=0.05, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--lambda_epochs', default=50, type=int)
    parser.add_argument('--ours_start_step', default=0, type=int)
    parser.add_argument('--lam', default=10, type=float, help="balance factor of kd_loss")
    parser.add_argument('--tau', default=0.5, type=float)
    parser.add_argument('--use_rl_eta', action='store_true', default=False)
    parser.add_argument('--ood', default=False, type=bool)  # If use ood strategy

    # args, unknown_args = parser.parse_known_args()
    args, unknown_args = parser.parse_known_args()

    # dataset_list = ['Scene15', 'Animal', 'BDGP', 'LandUse', 'Caltech101-20']   #
    dataset_list = ['animal']
    for dataset in dataset_list:
        args.data_name = dataset
        # config
        teacher_epochs_list, kd_epochs_list, d_model_list = [], [], []
        if args.data_name == 'Scene15':
            teacher_epochs_list = [100]
            kd_epochs_list = [100]
            d_model_list = [256]
            args.label_embedd = 256
            args.lam = 1
        elif args.data_name == 'animal':
            teacher_epochs_list = [100]
            kd_epochs_list = [100]
            args.label_embedd = 512
            args.lr = 0.0005
            args.lam = 1
            d_model_list = [512]
        elif args.data_name == 'Caltech101-20':
            teacher_epochs_list = [100]
            kd_epochs_list = [100]
            args.label_embedd = 512
            args.n_layers = 1
            args.lam = 1
            args.batch_size = 128
            d_model_list = [512]
        elif args.data_name == 'BDGP':
            teacher_epochs_list = [100]
            kd_epochs_list = [100]
            args.label_embedd = 512
            args.lr = 0.0005
            args.lam = 1
            d_model_list = [512]
        elif args.data_name == 'LandUse_21':
            teacher_epochs_list = [100]
            kd_epochs_list = [100]
            args.label_embedd = 512
            args.lr = 0.0005
            args.lam = 0.01
            args.tau = 0.7
            d_model_list = [512]

        if not os.path.exists('./result_F1'):
             os.mkdir('./result_F1')
        file_path = './result_F1/' + args.data_name + '.csv'
        # setup_seed(42)
        miss_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]   # 0.0, 0.1, 0.2, 0.3, 0.4, 0.5

        for mr in miss_rate:
            args.miss_rate = mr
            for Tepochs in teacher_epochs_list:
                args.teacher_epochs = Tepochs
                for Sepochs in kd_epochs_list:
                    args.kd_epochs = Sepochs
                    for dim in d_model_list:
                        args.d_model = dim
                        data_write_csv(file_path, '-----------------*****strategy 1*****-------------------')
                        ACC_set, F1_set = [], []
                        for i in range(0, 3):
                            print(i)
                            best_test_acc, best_test_f1, best_epoch = run(args, unknown_args, device, file_path)

                            data_write_csv(file_path, 'with_RL: ' + str(args.use_rl_eta) + ' miss_rate: '
                                           + str(args.miss_rate) + ' eta: ' + str(args.eta) + ' ood: ' + str(args.ood)
                                           + ' test--ACC: ' + str(f"{best_test_acc:.4f}") + ' test--F1: ' + str(f"{best_test_f1:.4f}")
                                           + ' best--epoch: ' + str(f"{best_epoch}"))

                            ACC_set.append(best_test_acc)
                            F1_set.append(best_test_f1)
                        ACC_set = torch.tensor(ACC_set)
                        F1_set = torch.tensor(F1_set)
                        meanACC = torch.mean(ACC_set)
                        stdACC = torch.std(ACC_set)
                        meanF1 = torch.mean(F1_set)
                        stdF1 = torch.std(F1_set)
                        data_write_csv(file_path, str(args))
                        data_write_csv(file_path, 'teacher_epochs: ' + str(args.teacher_epochs) + ' ' + 'kd_epochs: '
                                       + str(args.kd_epochs) + ' ' + 'lam: ' + str(args.lam) + ' ' + 'mean--ACC: ' + str(f"{meanACC:.4f}") + ' std--ACC:' + str(f"{stdACC:.4f}")
                                       + ' mean-F1: ' + str(f"{meanF1:.4f}") + ' std-F1: ' + str(f"{stdF1:.4f}"))


