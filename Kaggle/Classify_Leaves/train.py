# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2024/12/18 下午8:55
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from dataloader import Leaves_Dataset
from model import res_model


def train(model, train_dateloader, val_dataloader, loss_fn, optimizer, num_epoch, device: str = None,
          writer: SummaryWriter = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_val_acc = 0
    for epoch in tqdm(range(num_epoch), total=num_epoch):
        l_sum = 0
        for data in train_dateloader:
            optimizer.zero_grad()
            img_data, label_data = data
            label_hat = model(img_data.to(device))
            label_data = label_data.long()
            l = loss_fn(label_hat, label_data.to(device))
            l.backward()
            optimizer.step()
            l_sum += l.item()
        c_sum = 0
        with torch.no_grad():
            for val_data in val_dataloader:
                img_data, label_data = val_data
                label_hat = model(img_data.to(device))
                c = evaluate(label_hat.to('cpu'), label_data)
                c_sum += c
        acc = c_sum / len(val_dataloader.dataset)
        if writer is not None:
            writer.add_scalar('Loss/train', l_sum, epoch)
            writer.add_scalar('Accuracy/val', acc, epoch)

        torch.save(model.state_dict(), path_checkpoints / f"weights_last.pth")
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), path_checkpoints / f"weights_best.pth")
    if writer is not None:
        writer.close()


def evaluate(label_hat, label_data):
    label_hat = label_hat.argmax(dim=-1)
    correct = (label_hat == label_data).sum().item()
    return correct


if __name__ == '__main__':
    lr = 1e-3
    batch_size = 32
    num_epoch = 20

    path_logs = Path('logs')
    path_logs.mkdir(exist_ok=True)
    path_checkpoints = Path('checkpoints')
    path_checkpoints.mkdir(exist_ok=True)
    writer = SummaryWriter(str(path_logs))

    path_dateset = "classify-leaves"
    train_dataset = Leaves_Dataset(path_dateset, 'train')
    val_dataset = Leaves_Dataset(path_dateset, 'val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # train(model, train_dataloader, val_dataloader, )
    model = res_model(176)
    model.load_state_dict(torch.load(str(path_checkpoints / "weights_8.pth"), weights_only=True))

    param_1x = [param for name, param in model.named_parameters() if name not in ['fc.weight', 'fc.bias']]
    optimzer = torch.optim.Adam([{'params': param_1x}, {'params': model.fc.parameters(), 'lr': lr * 10}], lr=lr)

    loss_fn = torch.nn.CrossEntropyLoss()

    train(model, train_loader, val_loader, loss_fn, optimzer, num_epoch, writer=writer)
