# -*- coding: utf-8 -*-
import os
import argparse
import time
import torch
from Utils import HSIData, Toolkits, ResultContainer, PathManager, ModelTrainer, Plotkits, np
from model import TestNet

""" 
HyperParameters Setting
"""
cur_dir = os.path.dirname(os.path.abspath(__file__)) + '\\'
print("The current folder path is:", cur_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("The current device is:", device)

parser = argparse.ArgumentParser(description='PyTorch CLNet')
group_data = parser.add_argument_group('Dataset Setting')
group_data.add_argument('--data_path', type=str, default=cur_dir,
                        help='The path of the dataset (default is cur_dir)')
group_data.add_argument('--data_name', type=str, default='IP',
                        help='The name of the dataset: IP (default), PU, KSC, PC, SV, BS, HS2013')
group_data.add_argument('--if_norm', default=True, action='store_false',
                        help='Normalize the original data (default is True)')
group_data.add_argument('--ignored_labels', nargs='+', type=int, default=[0],
                        help='Ignored labels (default is 0)')

group_experiment = parser.add_argument_group('Experiment Setting')
group_experiment.add_argument('--save_path', type=str, default=cur_dir,
                              help='The path to save the results and records (default is cur_dir)')
group_experiment.add_argument('--num_iter', type=int, default=1,
                              help='Number of experiment repetitions (default to 1)')
group_experiment.add_argument('--sampling_seeds', nargs='+', type=int,
                              default=[1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340],
                              help='seeds for splitting the dataset (default is [1331, 1332, 1333, 1334, 1335, '
                                   '1336, 1337, 1338, 1339, 1340])')
group_experiment.add_argument('--train_ratio', type=float, default=0.1,
                              help='Sampling ratio of training (default to 0.1). '
                                   'Or sampling number per class of training (>=1).')
group_experiment.add_argument('--val_ratio', type=float, default=0.01,
                              help='Sampling ratio of val (default to 0.01).  '
                                   'If val_ratio is None, dataset only split into train set and test set.')

group_model = parser.add_argument_group('Model Setting')
group_model.add_argument('--device', type=str, default=device,
                         help="Specify CUDA device (defaults first to use cuda)")
group_model.add_argument('--seed', type=int, default=233,
                         help='Random seed for training')
group_model.add_argument('--lr', type=float, default=3e-4,
                         help="Learning rate, set by the model if not specified.")
group_model.add_argument('--l2_decay', type=float, default=2e-5,
                         help='the L2  weight decay')
group_model.add_argument('--num_epoch', type=int, default=1000,
                         help='The number of epoch')

args = parser.parse_args()
# args.num_epoch = 100
# args.num_iter = 2
if __name__ == '__main__':
    '''
    Data Preparing
    '''
    hsi = HSIData(current_dir=args.data_path, data_name=args.data_name)
    H, W, B = hsi.get_size()
    num_cls, num_total, count_cls = hsi.get_clsinfo(args.ignored_labels)
    result_container = ResultContainer(args.num_iter, num_cls)
    path_manager = PathManager(args.save_path, hsi.data_name, '11_18_20_24')


    for iter in range(args.num_iter):
        train_loader, test_loader, _ = hsi.get_dataloader(args.train_ratio, args.val_ratio,
                                                          seed=args.sampling_seeds[iter])
        Toolkits.seed_worker(args.seed)  # 设置随机种子
        '''
        Model Setting
        '''
        print("---------%d Iter--------" % (iter+1))
        # init model
        hyperparams = vars(args)
        hyperparams['num_band'] = B
        hyperparams['num_cls'] = num_cls
        trainer = ModelTrainer(TestNet, hyperparams)

        #full classification phase
        trainer.load_params(path_manager.get_dictpath(trainer.net, iter))
        _, test_pred, full_pred = trainer.evaluate(test_loader, True)
        #  map phase
        print('Generating ClsMap')
        Plotkits.draw_classification_map(
            path=path_manager.get_mappath(trainer.net, iter),
            label=hsi.gt,
            mask=test_loader.dataset[0][2],
            pred=test_pred
        )

        # full classification
        full_mask = train_loader.dataset[0][2]
        full_mask = ~full_mask
        Plotkits.draw_full_map(
            path=path_manager.get_mappath(trainer.net, iter, True),
            groundtruth=hsi.gt,
            full_mask=full_mask,
            full_pred=np.squeeze(full_pred, 0)
        )















