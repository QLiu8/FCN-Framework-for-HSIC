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

group_early_stop = parser.add_argument_group('EarlyStopping Setting')
group_early_stop.add_argument('--if_loss', type=bool, default=True,
                              help='Use val loss or val acc as the stopping criterion (default is loss)')
group_early_stop.add_argument('--patience', type=int, default=10,
                              help='How long to wait after last time val loss improved (default to 10)')
group_early_stop.add_argument('--delta', type=int, default=0,
                              help='Minimum change as an improvement (default to 0)')

args = parser.parse_args()
hyperparams = vars(args)

if __name__ == '__main__':
    '''
    Data Preparing
    '''
    hsi = HSIData(current_dir=hyperparams['data_path'], data_name=hyperparams['data_name'])
    H, W, B = hsi.get_size()
    num_cls, num_total, count_cls = hsi.get_clsinfo(hyperparams['ignored_labels'])
    result_container = ResultContainer(hyperparams['num_iter'], num_cls)
    path_manager = PathManager(hyperparams['save_path'], hsi.data_name)

    for iter in range(hyperparams['num_iter']):
        train_loader, test_loader, val_loader = hsi.get_dataloader(hyperparams['train_ratio'],
                                                                   hyperparams['val_ratio'],
                                                                   seed=hyperparams['sampling_seeds'][iter])
        print("---------%d Iter--------" % (iter + 1))
        print('training samples:', Toolkits.to_numpy(train_loader.dataset[0][2].sum()))
        print('val samples:', Toolkits.to_numpy(val_loader.dataset[0][2].sum()))
        print('test samples:', Toolkits.to_numpy(test_loader.dataset[0][2].sum()))
        Toolkits.seed_worker(hyperparams['seed'])  # 设置随机种子
        '''
        Model Setting
        '''
        # init model
        hyperparams = vars(args)
        hyperparams['num_band'] = B
        hyperparams['num_cls'] = num_cls
        trainer = ModelTrainer(TestNet, hyperparams)

        '''
        Model training
        '''
        # best_acc = -1
        # best_loss = 100
        loss_list = []
        print("training on ", hyperparams['device'])
        #training phase
        tic_train = time.perf_counter()
        for epoch in range(hyperparams['num_epoch']):
            train_acc, loss, val_acc, loss_val = trainer.train_per_epoch(train_loader, val_loader)

            loss_list.append(loss_val)
            # if best_acc < val_acc:
            #     trainer.save_params(path_manager.get_dictpath(trainer.net, iter))
            #     best_acc = val_acc
            # if best_loss > loss_val:
            #     trainer.save_params(path_manager.get_dictpath(trainer.net, iter))
            #     best_loss = loss_val
            early_stop = trainer.early_stop(save_path=path_manager.get_dictpath(trainer.net, iter),
                                            score_loss=loss_val,
                                            score_acc=val_acc,
                                            patience=hyperparams['patience'],
                                            delta=hyperparams['delta'],
                                            if_loss=hyperparams['if_loss'])

            if (epoch + 1) % 10 == 0:
                print("epoch %d, train_accuracy %g, train_loss %g, val_accuracy %g, val_loss %g" %
                      (epoch + 1, train_acc * 100, loss, val_acc * 100, loss_val))
            if early_stop:
                print("Early stopping")
                # 结束模型训练
                break

        toc_train = time.perf_counter()

        #testing phase
        tic_test = time.perf_counter()
        trainer.load_params(path_manager.get_dictpath(trainer.net, iter))
        test_labels, test_pred, full_pred = trainer.evaluate(test_loader, True)
        toc_test = time.perf_counter()

        """
        Statisticing Results
        """
        result_container.statistic_results(iter, test_labels, test_pred)
        result_container.calculate_time(iter, tic_train, toc_train)
        result_container.calculate_time(iter, tic_test, toc_test, False)
        print("OA: %g, AA: %g, Kappa: %g" %
              (result_container.overall_acc[iter],
               result_container.average_acc[iter],
               result_container.kappa[iter]))
        if iter+1 == hyperparams['num_iter']:
            result_container.record_result_xlsx(path_manager.get_recordpath(trainer.net, False))
            result_container.record_result_txt(path_manager.get_recordpath(trainer.net), hyperparams)
            print("Mean_OA: %g, Mean_AA: %g, Mean_Kappa: %g" %
                  (np.mean(result_container.overall_acc),
                   np.mean(result_container.average_acc),
                   np.mean(result_container.kappa)))
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













