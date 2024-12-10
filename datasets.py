import os
import sys

import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset    # , DataLoader

import util


def load_data(data_name):
    main_dir = sys.path[0]
    x_list = []
    y = None
    mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))

    if data_name in ['Fashion']:
        x_list.append(mat['X1'].reshape(10000, -1).astype('float32'))
        x_list.append(mat['X2'].reshape(10000, -1).astype('float32'))
        # x_list.append(mat['X3'].reshape(10000, -1).astype('float32'))
        y = np.squeeze(mat['Y'])

    elif data_name in ['BDGP']:
        x_list.append(mat['X1'].astype(np.float32))
        x_list.append(mat['X2'].astype(np.float32))
        y = np.squeeze(mat['Y'].transpose())

    elif data_name in ['HandWritten']:
        scaler = MinMaxScaler()
        x_all = mat['X'][0]
        for view in [0, 2]:     # range(x_all.shape[0])
            x_list.append(scaler.fit_transform(x_all[view].astype('float32')))
        y = np.squeeze(mat['Y']).astype('int')

    elif data_name in ['Reuters_dim10']:
        scaler = MinMaxScaler()
        x_train = mat['x_train']
        x_test = mat['x_test']
        y = np.squeeze(np.hstack((mat['y_train'], mat['y_test']))).astype('int')

        for view in [0, 1]:
            x_list.append(scaler.fit_transform(np.vstack((x_train[view], x_test[view])).astype('float32')))

    elif data_name in ['WebKB']:
        scaler = MinMaxScaler()
        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        x_list.append(scaler.fit_transform(mat['x1']).astype('float32'))
        x_list.append(scaler.fit_transform(mat['x2']).astype('float32'))
        y = np.squeeze(mat['y']).astype('int')

    elif data_name in ['Scene-15']:
        x_all = mat['X'][0]
        for view in [0, 1]:     # range(x_all.shape[0])
            x_list.append(x_all[view].astype('float32'))
        y = np.squeeze(mat['Y'])

    elif data_name in ['Caltech-2V', 'Caltech-3V', 'Caltech-4V', 'Caltech-5V']:
        scaler = MinMaxScaler()
        x_list.append(scaler.fit_transform(mat['X1'].astype(np.float32)))
        x_list.append(scaler.fit_transform(mat['X2'].astype(np.float32)))
        if data_name in ['Caltech-3V', 'Caltech-4V', 'Caltech-5V']:
            x_list.append(scaler.fit_transform(mat['X5'].astype(np.float32)))
        if data_name in ['Caltech-4V', 'Caltech-5V']:
            x_list.append(scaler.fit_transform(mat['X4'].astype(np.float32)))
        if data_name in ['Caltech-5V']:
            x_list.append(scaler.fit_transform(mat['X3'].astype(np.float32)))

        y = np.squeeze(mat['Y']).astype('int')

    elif data_name in ['Caltech101-7', 'Caltech101-20', 'Caltech101']:
        scaler = MinMaxScaler()
        x_all = mat['X'][0]
        for view in [3, 4]:  # range(x_all.shape[0]):
            # x_list.append(x_all[view].astype('float32'))
            x_list.append(scaler.fit_transform(x_all[view].astype('float32')))
        y = np.squeeze(mat['Y']).astype('int')

    elif data_name in ['aloideep3v', 'NH_face']:
        x_all = mat['X'][0]
        for view in range(x_all.shape[0]):
            x_list.append(x_all[view])
        y = np.squeeze(mat['truth']).astype('int')

    return x_list, y


def norm_data(data_name, x_list):
    if data_name in ['NH_face', 'aloideep3v', 'Reuters_dim10']:
        ss_list = [StandardScaler() for _ in range(len(x_list))]
        x_list_new = [ss_list[v].fit_transform(v_data.astype(np.float32)) for v, v_data in enumerate(x_list)]
    elif data_name in ['BDGP']:
        x_list_new = [util.normalize(x).astype('float32') for x in x_list]
    elif data_name in ['BDGP']:
        x_list_new = [util.normalize_row(x).astype('float32') for x in x_list]
    else:
        x_list_new = [x.astype('float32') for x in x_list]

    return x_list_new


def shuffle_data(x_list, aligned_ratio):
    num_views = len(x_list)  # 视图数量
    num_samples = x_list[0].shape[0]  # 样本数量

    # 计算对齐样本的数量
    num_aligned = int(num_samples * aligned_ratio)
    aligned_idx = np.append(np.ones(num_aligned), np.zeros(num_samples - num_aligned))
    np.random.shuffle(aligned_idx)

    # 初始化返回的对齐数据和打乱后的数据
    x_aligned = [x.copy() for x in x_list]  # 对齐的数据与输入相同
    x_shuffle = [x.copy() for x in x_list]  # 打乱后的数据

    # 获取需要打乱的样本索引
    misaligned_indices = np.where(aligned_idx == 0)[0]

    # 对除第一个视图外的视图，打乱 misaligned_indices 部分的样本
    for i in range(1, num_views):  # 保持第一个视图的顺序不变，从第二个视图开始打乱
        shuffle_indices = misaligned_indices.copy()
        np.random.shuffle(shuffle_indices)  # 随机打乱这些不对齐样本的顺序
        x_shuffle[i][misaligned_indices] = x_list[i][shuffle_indices]  # 打乱这些样本的顺序

    # aligned_idx表示样本是否对齐
    return x_aligned, x_shuffle, aligned_idx


class MvDataset(Dataset):
    def __init__(self, fea, labels, aligned_idx, device):
        self.device = device
        self.fea = fea
        self.labels = labels
        self.aligned_idx = aligned_idx

    def __getitem__(self, idx):
        return [torch.from_numpy(x[idx]).to(self.device) for x in self.fea], \
            self.labels[0][idx], \
            self.aligned_idx[idx], \
            torch.from_numpy(np.array(idx)).long()

    def __len__(self):
        return self.fea[0].shape[0]


def get_dataset(config, device):
    # load data from disk
    x, y = load_data(config['Dataset']['name'])

    # shuffle data
    x_aligned, x_shuffle, aligned_idx = shuffle_data(x, config['Dataset']['aligned_ratio'])

    # construct loader
    dataset_aligned = MvDataset(x_aligned, [y, y], aligned_idx, device)
    dataset_shuffle = MvDataset(x_shuffle, [y, y], aligned_idx, device)
    # data_loader = DataLoader(
    #     dataset,
    #     batch_size=config['Dataset']['batch_size'],
    #     shuffle=True
    # )

    return dataset_aligned, dataset_shuffle, aligned_idx
