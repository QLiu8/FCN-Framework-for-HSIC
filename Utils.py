# -*- coding: utf-8 -*-
"""
This file contains the dataset for hyperspectral images and related helpers.
"""
import os
import copy
import math
import torch
import random
import datetime
import h5py
import numpy as np
import torch.optim as optim
import pandas as pd
import scipy.io as sio
import spectral as spy
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import torch.utils.data as Data
from operator import truediv
from sklearn import metrics




class HSIData:
    """
    dataname: IP,PU,KSC,PC,SV,BS,HS2013
    """
    def __init__(self, current_dir='.\\', data_name='IP', if_norm=True):

        self.__cur_dir = current_dir
        self.data_name = data_name
        self.__if_norm = if_norm
        if self.data_name == 'IP':
            mat_data = sio.loadmat(self.__cur_dir+'datasets\\IndianP\\Indian_pines_corrected.mat')
            self.data = mat_data['indian_pines_corrected']
            mat_gt = sio.loadmat(self.__cur_dir+'datasets\\IndianP\\Indian_pines_gt.mat')
            self.gt = mat_gt['indian_pines_gt']
            # label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
            #                 "Corn", "Grass-pasture", "Grass-trees",
            #                 "Grass-pasture-mowed", "Hay-windrowed", "Oats",
            #                 "Soybean-notill", "Soybean-mintill", "Soybean-clean",
            #                 "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
            #                 "Stone-Steel-Towers"]
        elif self.data_name == 'PU':
            mat_data = sio.loadmat(self.__cur_dir+'datasets\\PaviaU\\PaviaU.mat')
            self.data = mat_data['paviaU']
            mat_gt = sio.loadmat(self.__cur_dir+'datasets\\PaviaU\\PaviaU_gt.mat')
            self.gt = mat_gt['paviaU_gt']
            # label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
            #                 'Painted metal sheets', 'Bare Soil', 'Bitumen',
            #                 'Self-Blocking Bricks', 'Shadows']
        elif self.data_name == 'KSC':
            mat_data = sio.loadmat(self.__cur_dir+'datasets\\KSC\\KSC.mat')
            self.data = mat_data['KSC']
            mat_gt = sio.loadmat(self.__cur_dir+'datasets\\KSC\\KSC_gt.mat')
            self.gt = mat_gt['KSC_gt']
            # label_values = ["Undefined", "Scrub", "Willow swamp",
            #                 "Cabbage palm hammock", "Cabbage palm\\oak hammock",
            #                 "Slash pine", "Oak\\broadleaf hammock",
            #                 "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
            #                 "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]
        elif self.data_name == 'PC':
            mat_data = sio.loadmat(self.__cur_dir+'datasets\\PaviaC\\Pavia.mat')
            self.data = mat_data['pavia']
            mat_gt = sio.loadmat(self.__cur_dir+'datasets\\PaviaC\\Pavia_gt.mat')
            self.gt = mat_gt['pavia_gt']
            # label_values = ["Undefined", "Water", "Trees", "Asphalt",
            #                 "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
            #                 "Meadows", "Bare Soil"]
        elif self.data_name == 'SV':
            mat_data = sio.loadmat(self.__cur_dir+'datasets\\Salinas\\Salinas_corrected.mat')
            self.data = mat_data['salinas_corrected']
            mat_gt = sio.loadmat(self.__cur_dir+'datasets\\Salinas\\Salinas_gt.mat')
            self.gt = mat_gt['salinas_gt']
            # label_values = ["Brocoli_green_weeds_1", "Brocoli_green_weeds_2 Fallow", "Fallow_rough_plow", "Fallow_smooth",
            #                 "Stubble", "Celery", "Grapes_untrained", "Soil_vinyard_develop", "Corn_senesced_green_weeds",
            #                 "Lettuce_romaine_4wk", "Lettuce_romaine_5wk", "Lettuce_romaine_6wk", "Lettuce_romaine_7wk",
            #                 "Vinyard_untrained", "Vinyard_vertical_trellis"]
        elif self.data_name == 'BS':
            mat_data = sio.loadmat(self.__cur_dir+'datasets\\Botswana\\Botswana.mat')
            self.data = mat_data['Botswana']
            mat_gt = sio.loadmat(self.__cur_dir+'datasets\\Botswana\\Botswana_gt.mat')
            self.gt = mat_gt['Botswana_gt']
            # label_values = ["Undefined", "Water", "Hippo grass",
            #                 "Floodplain grasses 1", "Floodplain grasses 2",
            #                 "Reeds", "Riparian", "Firescar", "Island interior",
            #                 "Acacia woodlands", "Acacia shrublands",
            #                 "Acacia grasslands", "Short mopane", "Mixed mopane",
            #                 "Exposed soils"]
        elif self.data_name == 'HS2013':
            mat_data = sio.loadmat(self.__cur_dir+'datasets\\Houston2013\\Houston.mat')
            self.data = mat_data['Houston']
            mat_gt = sio.loadmat(self.__cur_dir+'datasets\\Houston2013\\Houston_gt.mat')
            self.gt = mat_gt['Houston_gt']
            # label_values = ["Undefined", "Healthy grass", "Stressed grass", "Synthetic grass",
            #                 "Trees", "Soil", "Water", "Residential", "Commercial", "Road",
            #                 "Highway", "Railway", "Parking Lot 1", "Parking Lot 2", "Tennis Court",
            #                 "Running Track"]
        elif self.data_name == 'HS2018':
            mat_data = h5py.File(self.__cur_dir + 'datasets\\Houston2018\\HoustonU.mat')
            self.data = mat_data['houstonU'][()]
            mat_gt = h5py.File(self.__cur_dir + 'datasets\\Houston2018\\HoustonU_gt.mat')
            self.gt = mat_gt['houstonU_gt'][()]
        try:
            if self.__if_norm:
                self.data = self.__normalize_data()
            else:
                self.data = self.data.astype(np.float32)
        except:
            print("dataname only includes IP,PU,KSC,PC,SV,BS,HS2013")
        self.__H, self.__W, self.__B = self.data.shape

    def __normalize_data(self):
        img2d = self.data.reshape(np.prod(self.data.shape[:2]), np.prod(self.data.shape[2:]))  # 二维展开之后的高光谱数据
        minMax = preprocessing.StandardScaler()
        img2d = minMax.fit_transform(img2d)
        self.data = img2d.reshape(self.data.shape[0], self.data.shape[1], self.data.shape[2])
        print('Complete the normalization of the original data.')
        return self.data
    def get_size(self):
        """
        get data size of hsi
        :return: Height, width and band
        """
        return self.__H, self.__W, self.__B
    def get_clsinfo(self, ignored_labels=[0]):
        """
        获得数据集信息(不包含ignored_labels)：类别总数，标记样本总数，每类样本数，修改后的gt(2D)
        :param ignored_labels: 排除在外的类别，0为无标记数据，默认排除
        :return: num_cls, num_total, count_cls, 修改self.gt
        """
        gt1d = self.gt.reshape(self.__H*self.__W,)
        unique_gt = np.unique(gt1d)
        print('Class labels of the original data are:', unique_gt)
        print('Ignored labels are:', ignored_labels)
        if ignored_labels == [0]:
            pass
        else:
            for i in ignored_labels:
                if i != 0:
                    indexes = [j for j, x in enumerate(gt1d.ravel().tolist()) if x == i]
                    gt1d[indexes] = 0
            unique_gt = np.unique(gt1d)
            if max(unique_gt) > len(unique_gt) - 1:
                for i in range(len(unique_gt) - 1):
                    indexes = [j for j, x in enumerate(gt1d.ravel().tolist()) if x == unique_gt[i + 1]]
                    gt1d[indexes] = i + 1
            self.gt = gt1d.reshape(self.__H, self.__W) #update gt to exclude ignored labels
        assert max(gt1d) == len(unique_gt)-1
        cls, count_cls = np.unique(gt1d, return_counts=True)
        self.__num_cls = len(cls)-1      #0：background
        count_cls = count_cls[1:]
        num_total = np.sum(count_cls)
        print('Data classes after excluding ignored labels are:', cls)
        return self.__num_cls, num_total, count_cls
    def to_dataloader(self, indices):
        """
        generate dataloader with indices， including data, gt, mask,
        :param self:
        :param indices: indexes of samples
        :return: dataloader for tensor
        """
        gt = self.__get_gt(indices) - 1
        Data_Tensor = torch.from_numpy(self.data).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(0)
        GT_Tensor = torch.from_numpy(gt).type(torch.FloatTensor).unsqueeze(0)
        Mask_Tensor = GT_Tensor.ge(0)
        tensordataset = Data.TensorDataset(Data_Tensor, GT_Tensor, Mask_Tensor)
        dataloader = Data.DataLoader(tensordataset)
        return dataloader
    def __get_gt(self, indices):#生成gt大小的2d矩阵，索引位置保留label
        gt = self.gt.reshape(self.__H*self.__W)
        gt_new = np.zeros(len(gt))
        gt_new[indices] = gt[indices]
        gt_new = gt_new.reshape(self.__H, self.__W)
        return np.array(gt_new, dtype='int8')
    def get_dataloader(self, train_ratio, val_ratio=None, minimun=1, seed=None):
        """
        Split the dataset and generate dataloader with indices
        :param train_ratio:
        :param val_ratio:
        :param minimun:
        :param seed: np random seed
        :return:
        """

        if val_ratio==None:#only train and test
            if train_ratio >= 1:  # fix the number of samples per class
                train_idx, test_idx = Toolkits.sampling_per_class(self.gt, train_ratio, val_ratio,
                                                                              minimun, seed)
            else:
                train_idx, test_idx = Toolkits.sampling(self.gt, train_ratio, val_ratio,
                                                        minimun, seed)
            train_loader = self.to_dataloader(train_idx)
            test_loader = self.to_dataloader(test_idx)
            return train_loader, test_loader
        else:# add val
            if train_ratio >= 1:  # fix the number of samples per class
                train_idx, test_idx, val_idx = Toolkits.sampling_per_class(self.gt, train_ratio, val_ratio,
                                                                           minimun, seed)
            else:#fix the sampling ratio for each class
                train_idx, test_idx, val_idx = Toolkits.sampling(self.gt, train_ratio, val_ratio,
                                                                 minimun, seed)
            train_loader = self.to_dataloader(train_idx)
            test_loader = self.to_dataloader(test_idx)
            val_loader = self.to_dataloader(val_idx)
            return train_loader, test_loader, val_loader

class Toolkits:
    @staticmethod
    def sampling(ground_truth, training_ratio, val_ratio=None, minimum=1, seed=None):
        """
        按每类样本比例划分训练、验证、测试
        :param training_ratio:<1
        :param val_ratio:<1 or None(不划分val)
        :param ground_truth:2D array, 0 为背景
        :param minimum: 每类样本的最小采样数
        :return: train_indexes, val_indexes, test_indexes
        """
        if seed == None:
            pass
        else:
            np.random.seed(seed)  # Numpy module

        train = {}
        test = {}
        val = {}
        ground_truth = ground_truth.reshape(np.prod(ground_truth.shape[:2]), )
        m = max(ground_truth).astype(int)
        for i in range(m):
            indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]
            np.random.shuffle(indexes)
            num_training = math.ceil(training_ratio * len(indexes))
            if num_training < minimum:
                num_training = minimum
            train[i] = indexes[: num_training]
            if val_ratio == None:
                test[i] = indexes[num_training:]
            else:
                num_val = math.ceil(val_ratio * len(indexes))
                if num_val < minimum:
                    num_val = minimum
                val[i] = indexes[num_training: num_training + num_val]
                test[i] = indexes[num_training + num_val:]
        train_indexes = []
        test_indexes = []
        val_indexes = []
        for i in range(m):
            train_indexes += train[i]
            test_indexes += test[i]
            if val_ratio == None:
                pass
            else:
                val_indexes += val[i]
        np.random.shuffle(train_indexes)
        np.random.shuffle(test_indexes)
        if val_ratio == None:
            return train_indexes, test_indexes
        else:
            np.random.shuffle(val_indexes)
            return train_indexes, test_indexes, val_indexes

    @staticmethod
    def sampling_per_class(ground_truth, training_per_class, val_per_class=None, min_ratio=0.5, seed=None):
        if seed == None:
            pass
        else:
            np.random.seed(seed)  # Numpy module
        train = {}
        test = {}
        val = {}
        ground_truth = ground_truth.reshape(np.prod(ground_truth.shape[:2]), )
        m = max(ground_truth).astype(int)
        for i in range(m):
            indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]
            np.random.shuffle(indexes)
            training_per_class = int(training_per_class)
            if val_per_class == None:
                if len(indexes) * min_ratio < training_per_class:
                    nb_train = round(len(indexes) * min_ratio)
                    train[i] = indexes[:nb_train]
                    test[i] = indexes[nb_train:]
                else:
                    train[i] = indexes[: training_per_class]
                    test[i] = indexes[training_per_class:]
            else:
                val_per_class = int(val_per_class)
                if len(indexes) * min_ratio < training_per_class:
                    nb_train = round(len(indexes) * min_ratio)
                    train[i] = indexes[:nb_train]
                    val[i] = indexes[nb_train: nb_train + val_per_class]
                    test[i] = indexes[nb_train + val_per_class:]
                else:
                    train[i] = indexes[: training_per_class]
                    val[i] = indexes[training_per_class: training_per_class + val_per_class]
                    test[i] = indexes[training_per_class + val_per_class:]
        train_indexes = []
        val_indexes = []
        test_indexes = []
        for i in range(m):
            train_indexes += train[i]
            test_indexes += test[i]
            if val_per_class == None:
                pass
            else:
                val_indexes += val[i]
        np.random.shuffle(train_indexes)
        np.random.shuffle(test_indexes)
        if val_per_class == None:
            return train_indexes, test_indexes
        else:
            np.random.shuffle(val_indexes)
            return train_indexes, test_indexes, val_indexes

    @staticmethod
    def seed_worker(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if multi-GPU
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def get_param_num(model):
        num = sum(x.numel() for x in model.parameters())
        print("model has {} parameters in total".format(num))
        return num

    @staticmethod
    def to_numpy(data_tensor):
        return data_tensor.cpu().detach().numpy()

    @staticmethod
    def get_daytime():
        day = datetime.datetime.now()
        return day.strftime('%m_%d_%H_%M')

    @staticmethod
    def path_check(path):
        if not os.path.exists(path):  # 如果路径不存在
            os.makedirs(path)
            # print(path, 'is generated.')
        else:
            # print(path, 'already exists.')
            pass

class PathManager:
    def __init__(self, root, dataname, daytime=None):
        if daytime is None:
            self.__daytime = Toolkits.get_daytime()
        else:
            self.__daytime = daytime
        self.__file_name = dataname + '_' + self.__daytime

        self.__model_path = root + 'models\\'
        Toolkits.path_check(self.__model_path)
        self.__result_path = root + 'results\\'
        Toolkits.path_check(self.__result_path)
        self.__map_path = root + 'maps\\'
        Toolkits.path_check(self.__map_path)

    def get_dictpath(self, net, iter):
        self.dictpath = self.__model_path + net.name + '_' + self.__file_name + '\\'
        Toolkits.path_check(self.dictpath)  # 文件保存路径，如果不存在就会被重建
        return self.dictpath + 'params_' + str(iter + 1) + '.pkl'

    def get_recordpath(self, net, if_txt=True):
        self.recordpath = self.__result_path + net.name + '_' + self.__file_name + '\\'
        Toolkits.path_check(self.recordpath)
        if if_txt:
            return self.recordpath + 'results.txt'
        else:
            return self.recordpath + 'results.xlsx'

    def get_mappath(self, net, iter=None, if_full=False):
        self.clsmap_path = self.__map_path + net.name + '_' + self.__file_name + '\\'
        Toolkits.path_check(self.clsmap_path)
        if iter is None:
            return self.clsmap_path + 'GT.png'
        elif if_full:
            return self.clsmap_path + 'full_' + str(iter + 1) + '.png'
        else:
            return self.clsmap_path + 'only_' + str(iter + 1) + '.png'


class ResultContainer:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, ex_iter, num_cls):
        self.class_acc, self.overall_acc, self.average_acc, self.kappa, self.train_time, self.test_time = \
            (np.zeros((ex_iter, num_cls)), np.zeros(ex_iter), np.zeros(ex_iter), np.zeros(ex_iter),
             np.zeros(ex_iter), np.zeros(ex_iter))
        self.confusion_matrix = np.zeros((ex_iter, num_cls, num_cls))

    def statistic_results(self, iter, true, pred):
        self.overall_acc[iter] = metrics.accuracy_score(true, pred)
        self.confusion_matrix[iter, :, :] = metrics.confusion_matrix(true, pred)
        list_diag = np.diag(self.confusion_matrix[iter, :, :])
        list_raw_sum = np.sum(self.confusion_matrix[iter, :, :], axis=1)
        self.class_acc[iter, :] = np.nan_to_num(truediv(list_diag, list_raw_sum))
        self.average_acc[iter] = np.mean(self.class_acc)
        self.kappa[iter] = metrics.cohen_kappa_score(true, pred)
        pass

    def calculate_time(self, iter, tic, toc, if_train=True):
        if if_train:
            self.train_time[iter] = toc - tic
        else:
            self.test_time[iter] = toc - tic
        pass

    def record_result_txt(self, path, hyperparam):
        f = open(path, 'a')
        sentence0 = 'OAs for each iteration are: ['
        sentence1 = 'AAs for each iteration are: ['
        sentence2 = 'KAPPAs for each iteration are: ['
        for i in range(len(self.overall_acc)):
            sentence0 = sentence0 + str(round(self.overall_acc[i] * 100, 2)) + ' '
            sentence1 = sentence1 + str(round(self.average_acc[i] * 100, 2)) + ' '
            sentence2 = sentence2 + str(round(self.kappa[i] * 100, 2)) + ' '
        sentence0 = sentence0 + ']\n'
        sentence1 = sentence1 + ']\n'
        sentence2 = sentence2 + ']\n'
        f.write(sentence0)
        f.write(sentence1)
        f.write(sentence2)
        sentence3 = 'mean_OA ± std_OA is: ' + str(round(np.mean(self.overall_acc) * 100, 2)) + '±' + str(
            round(np.std(self.overall_acc) * 100, 2)) + '\n'
        f.write(sentence3)
        sentence4 = 'mean_AA ± std_AA is: ' + str(round(np.mean(self.average_acc) * 100, 2)) + '±' + str(
            round(np.std(self.average_acc) * 100, 2)) + '\n'
        f.write(sentence4)
        sentence5 = 'mean_KAPPA ± std_KAPPA is: ' + str(round(np.mean(self.kappa) * 100, 2)) + '±' + str(
            round(np.std(self.kappa) * 100, 2)) + '\n'
        f.write(sentence5)
        sentence6 = 'Total average Training time is: ' + str(round(np.mean(self.train_time), 2)) + '±' + str(
            round(np.std(self.train_time), 2)) + '\n'
        f.write(sentence6)
        sentence7 = 'Total average Testing time is: ' + str(round(np.mean(self.test_time), 2)) + '±' + str(
            round(np.std(self.test_time), 2)) + '\n' + '\n'
        f.write(sentence7)

        element_mean = np.mean(self.class_acc, axis=0)
        element_std = np.std(self.class_acc, axis=0)
        sentence8 = "Mean ± std of all elements in confusion matrix: " + '\n'

        for i in range(self.class_acc.shape[1]):
            sentence8 = sentence8 + str(round(element_mean[i] * 100, 2)) + '±' + str(
                round(element_std[i] * 100, 2)) + '\n'


        f.write(sentence8)
        sentence10 = "---------------------------------" + '\n'
        f.write(sentence10)
        for key, value in hyperparam.items():
            f.write(f"{key}: {value}\n")
        f.close()
        pass

    def record_result_xlsx(self, path):
        data = {}
        for i in range(self.class_acc.shape[1]):
            key = "class " + str(i + 1)
            data[key] = self.__add_mean_std(self.class_acc[:, i] * 100)

        data["oa"] = self.__add_mean_std(self.overall_acc * 100)
        data["aa"] = self.__add_mean_std(self.average_acc * 100)
        data["kappa"] = self.__add_mean_std(self.kappa * 100)
        data["train time"] = self.__add_mean_std(self.train_time)
        data["test time"] = self.__add_mean_std(self.test_time)

        df = pd.DataFrame(data).T
        writer = pd.ExcelWriter(path)
        df.to_excel(writer, sheet_name='sheet1')
        writer._save()

    def __add_mean_std(self, array):
        mean_std = str(round(np.mean(array), 2)) + '±' + str(round(np.std(array), 2))
        array = np.around(array, 2).astype(str)
        return np.append(array, mean_std)

class ModelTrainer:
    def __init__(self, model, hyperparams):
        self.net = model(hyperparams['num_band'], hyperparams['num_cls']).to(hyperparams['device'])
        self.num_params = Toolkits.get_param_num(self.net)
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = optim.Adam(self.net.parameters(), lr=hyperparams['lr'], weight_decay=hyperparams['l2_decay'])
        self.device = hyperparams['device']
        #early_stop setting
        self.counter = 0
        self.best_score = None

    def train_per_epoch(self, train_loader, val_loader):
        # training phase
        for idx_batch, (inputs, targets, mask) in enumerate(train_loader):
            self.net.train()
            inputs, targets, mask = inputs.to(self.device), targets.to(self.device), mask.to(self.device)
            outputs = self.net(inputs)
            loss = self.loss_func(outputs, targets.long())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_pred = Toolkits.to_numpy(outputs.argmax(dim=1)[mask])
            train_labels = Toolkits.to_numpy(targets[mask])
            train_acc = metrics.accuracy_score(train_labels, train_pred)
            # evaluating phase
            val_acc, loss_val = self.evaluate(val_loader)
        return train_acc, loss, val_acc, loss_val

    def evaluate(self, val_loader, if_test=False):
        with torch.no_grad():
            self.net.eval()
            for idx_val_batch, (inputs_val, targets_val, mask_val) in enumerate(val_loader):
                inputs_val, targets_val, mask_val = (inputs_val.to(self.device),
                                                     targets_val.to(self.device),
                                                     mask_val.to(self.device))
                outputs_val = self.net(inputs_val)
                loss_val = self.loss_func(outputs_val, targets_val.long())  # + lw*criterion_2(inputs, outputs_spe)

                val_pred = Toolkits.to_numpy(outputs_val.argmax(dim=1)[mask_val])
                val_labels = Toolkits.to_numpy(targets_val[mask_val])
        if not if_test:
            val_acc = metrics.accuracy_score(val_labels, val_pred)
            return val_acc, loss_val
        else:
            full_pred = Toolkits.to_numpy(outputs_val.argmax(dim=1))
            return val_labels, val_pred, full_pred

    def save_params(self, path):
        torch.save(self.net.state_dict(), path)

    def load_params(self, path):
        self.net.load_state_dict(torch.load(path))

    def early_stop(self, save_path, score_loss, score_acc, patience=10, delta=0, if_loss=True):
        if if_loss:# save best params according to val loss
            if self.best_score is None:
                self.best_score = score_loss
                self.save_params(save_path)
            elif score_loss > self.best_score + delta:#loss increase
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {patience}')
                if self.counter >= patience:
                    return True
            else:
                self.counter = 0
                self.best_score = score_loss
                self.save_params(save_path)
        else:
            if self.best_score is None:
                self.best_score = score_acc
                self.save_params(save_path)
            elif score_acc < self.best_score + delta:#acc decrease
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {patience}')
                if self.counter >= patience:
                    return True
            else:
                self.counter = 0
                self.best_score = score_acc
                self.save_params(save_path)
class Plotkits:

    @staticmethod
    def draw_classification_map(path, groundtruth, mask=None, pred=None, colors=None, scale: float = 4.0, dpi: int = 400):
        '''
        get classification map , then save to given path
        '''
        label = copy.copy(groundtruth)
        try:
            if mask is None and pred is None:
                pass
            else:
                if type(mask) != np.ndarray:
                    mask = Toolkits.to_numpy(mask)
                label[mask] = pred + 1
        except:
            print('Missing mask or pred')

        fig, ax = plt.subplots()
        spy.imshow(classes=label.astype(np.int16), fignum=fig.number, colors=colors)
        ax.set_axis_off()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
        foo_fig = plt.gcf()  # 'get current figure'
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        foo_fig.savefig(path, format='png', transparent=True, dpi=dpi, pad_inches=0)
        return None
    @staticmethod
    def draw_full_map(path, groundtruth, full_pred, full_mask, colors=None, scale: float = 4.0, dpi: int = 400):
        label = copy.copy(groundtruth)
        label[full_mask] = full_pred[full_mask] + 1
        fig, ax = plt.subplots()
        spy.imshow(classes=label.astype(np.int16), fignum=fig.number, colors=colors)
        ax.set_axis_off()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
        foo_fig = plt.gcf()  # 'get current figure'
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        foo_fig.savefig(path, format='png', transparent=True, dpi=dpi, pad_inches=0)
        pass


if __name__ == '__main__':
    Toolkits.path_check('./model/map/result')
    hsi = HSIData(data_name='PU')
    train_loader, _ = hsi.get_dataloader(20)
    pred = np.ones(180)
    Plotkits.draw_classification_map('test3.png', hsi.gt, train_loader.dataset[0][2])

    pass


