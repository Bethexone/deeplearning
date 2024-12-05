# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2024/11/12 下午5:23
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold


def dataset_process(path_train, path_test):
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)
    all_features = pd.concat([df_train.iloc[:, 1:-1], df_test.iloc[:, 1:]], axis=0)
    numeric_features = all_features.select_dtypes(include='number')
    numeric_features = numeric_features.apply(lambda x: (x - x.mean()) / x.std())
    numeric_features = numeric_features.fillna(0)

    categorical_features = all_features.select_dtypes(include='object')
    categorical_features = pd.get_dummies(categorical_features, dummy_na=True)

    all_features = pd.concat([numeric_features, categorical_features], axis=1)
    np_all_features = all_features.to_numpy()
    np_price = df_train.iloc[:, -1].to_numpy()
    np_train = np_all_features[:len(df_train), :].astype('float32')
    # np_train = np.concatenate((np_train, np_price[:, np.newaxis]), axis=1)
    np_test = np_all_features[len(df_train):, :].astype('float32')

    # print("success !")
    return np_train, np_price, np_test


class HousePrice_TrainDataset(Dataset):
    def __init__(self, path_train, path_test):
        self.np_train, self.np_price, _ = dataset_process(path_train, path_test)

    def __len__(self):
        return len(self.np_train)

    def __getitem__(self, idx):
        return torch.tensor(self.np_train[idx]), torch.tensor(self.np_price[idx], dtype=torch.float32)


class HousePrice_Datasetloader():
    def __init__(self, train_dataset, k_splits=10, batch_size=32):
        self.train_loader = None
        self.val_loader = None
        self.current_fold = 0
        self.k_splits = k_splits
        self.kf = KFold(n_splits=self.k_splits, shuffle=True)
        self.dataset_train = train_dataset
        self.fold_iter = self.kf.split(self.dataset_train)
        self.batch_size = batch_size

    def initialize(self):
        self.train_loader = None
        self.val_loader = None
        self.current_fold = 0
        self.kf = KFold(n_splits=self.k_splits, shuffle=True)
        self.fold_iter = self.kf.split(self.dataset_train)

    def update_loader(self):
        train_idx, val_idx = next(self.fold_iter)

        # 根据索引划分训练集和验证集
        train_subset = torch.utils.data.Subset(self.dataset_train, train_idx)
        val_subset = torch.utils.data.Subset(self.dataset_train, val_idx)

        # 创建 DataLoader
        self.train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_subset, batch_size=len(val_idx), shuffle=False)

        # 重置当前折的训练数据迭代器
        self.train_iter = iter(self.train_loader)
        self.val_iter = iter(self.val_loader)

        self.val_batch = next(self.val_iter)

    def __iter__(self):
        return self

    def __next__(self):
        if self.train_loader is None or self.val_loader is None:
            self.update_loader()

        try:
            # 获取当前训练和验证集的一个批次
            train_batch = next(self.train_iter)
        except StopIteration:
            # 如果当前折的数据已经遍历完，更新为下一个折
            self.current_fold += 1
            if self.current_fold < self.k_splits:
                self.update_loader()
                train_batch = next(self.train_iter)
            else:
                self.initialize()
                raise StopIteration  # 所有折数遍历完毕

        return train_batch, self.val_batch


if __name__ == '__main__':
    path_train = 'house-prices-advanced-regression-techniques/train.csv'
    path_test = 'house-prices-advanced-regression-techniques/test.csv'
    np_train, np_price, np_test = dataset_process(path_train, path_test)
    traindataset = HousePrice_TrainDataset(path_train, path_test)
    train_datasetloader = HousePrice_Datasetloader(traindataset)
    num_epoch = 10
    for i in range(num_epoch):
        for j, batch in enumerate(train_datasetloader):
            print(j)
    print("")
