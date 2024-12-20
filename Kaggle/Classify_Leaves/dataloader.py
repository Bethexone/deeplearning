# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2024/12/17 下午8:42
import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from PIL import Image
from torchvision import transforms


class Leaves_Dataset(data.Dataset):
    def __init__(self, path_dateset, mode: str = 'train'):
        super().__init__()
        self.path_dateset = Path(path_dateset)
        self.mode = mode
        path_train_csv = self.path_dateset / 'train.csv'
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整到固定大小
            transforms.ToTensor(),  # 转换为 Tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        df_dataset = pd.read_csv(path_train_csv)
        self.label_encoder = LabelEncoder()
        df_dataset["category"] = self.label_encoder.fit_transform(df_dataset["label"])

        train_data, val_data = train_test_split(df_dataset, test_size=0.2, random_state=42)

        match mode:
            case 'train':
                self.df_train = train_data
                self.df_data = self.df_train
            case 'val':
                self.df_val = val_data
                self.df_data = self.df_val
            case 'test':
                path_test_csv = Path(path_dateset) / 'test.csv'
                self.df_test = pd.read_csv(path_test_csv)
                self.df_data = self.df_test

    def __getitem__(self, index):
        row = self.df_data.iloc[index]
        image_path = self.path_dateset / row["image"]  # 获取 "images/5.jpg"
        img = Image.open(image_path)
        img = self.transform(img)
        if self.mode == 'test':
            return img
        else:
            label = row["category"]  # 获取 "ulmus_rubra"
            return img, label

    def __len__(self):
        return len(self.df_data)


if __name__ == '__main__':
    path_dateset = "classify-leaves"
    csv_path = Path(path_dateset) / "train.csv"
    df_dataset = pd.read_csv(csv_path)
    mode = 'test'
    traindataset = Leaves_Dataset(path_dateset,mode)
    Dataloader = data.DataLoader(traindataset, batch_size=32, shuffle=True)
    for data in Dataloader:
        img, label = data
        print(label)
        break
