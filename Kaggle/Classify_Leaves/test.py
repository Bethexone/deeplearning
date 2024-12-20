# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2024/12/19 下午1:30
import pandas as pd
import torch
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from model import res_model
from dataloader import Leaves_Dataset
from datetime import datetime


def test(model, test_loader: DataLoader, device: str = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = []
    model.eval()
    model.to(device)
    for data in tqdm(test_loader, total=len(test_loader)):
        label_hat = model(data.to(device))
        label = label_hat.argmax(dim=-1).to('cpu')
        result += label.tolist()


    label_txt = test_loader.dataset.label_encoder.inverse_transform(result)
    test_data = pd.read_csv(path_datasets / 'test.csv')
    test_data['label'] = label_txt
    test_data.to_csv(path_results / f'results_{current_date}.csv', index=False)


if __name__ == '__main__':
    current_date = datetime.now().strftime('%d_%m')
    path_checkpoints = Path('checkpoints')
    path_datasets = Path('classify-leaves')
    path_results = Path('results')
    path_results.mkdir(exist_ok=True)
    model = res_model(176)
    model.load_state_dict(torch.load(str(path_checkpoints / "weights_best.pth"), weights_only=True))
    test_dataset = Leaves_Dataset(path_datasets, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test(model, test_dataloader)
