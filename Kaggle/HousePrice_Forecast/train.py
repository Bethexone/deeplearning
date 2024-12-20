# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2024/11/13 上午10:52
import os.path

from dataset_load import *
from model import *
from torch.utils.tensorboard import SummaryWriter


def train(train_datasetloader, net, loss, optimizer, num_epoch, data_vis, device="cuda"):
    global val_batch

    def init_weights(layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)

    net.apply(init_weights)
    net.to(device)
    l_sum = 0
    val_sum = 0
    for epoch in range(num_epoch):
        for train_batch, val_batch in train_datasetloader:
            # train_batch = train_batch.cuda()
            # val_batch = val_batch.cuda()
            train_data, train_target = train_batch

            optimizer.zero_grad()
            output = net(train_data.cuda())
            l = loss(output, train_target.cuda())
            l.backward()
            optimizer.step()
            l_sum += l

        val_data, val_target = val_batch
        val_output = net(val_data.cuda())
        val_l = loss(val_output, val_target.cuda())
        val_sum += val_l.item()
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epoch}], Loss: {l_sum:.4f}, val_Loss: {val_l:.4f}')
            data_vis.add_scalar('Loss/train', l_sum, epoch)
            data_vis.add_scalar('Loss/val', val_l, epoch)

            l_sum = 0
            val_sum = 0


def pred(net, path_pth, test_features, device='cuda'):
    test_data = pd.read_csv(path_test)

    net.load_state_dict(torch.load(path_pth, weights_only=True))
    net.eval()
    test_features = torch.tensor(test_features).float().to(device)
    net.to(device)
    with torch.no_grad():
        test_pred = net(test_features)
    test_pred = test_pred.cpu().detach().numpy()
    test_result = pd.DataFrame()
    test_result['SalePrice'] = pd.Series(test_pred.reshape(-1))
    submission = pd.concat([test_data['Id'], test_result['SalePrice']], axis=1)
    submission.to_csv(f'{root_pred}/{os.path.splitext(os.path.basename(path_pth))[0]}.csv', index=False)


if __name__ == '__main__':
    path_train = 'house-prices-advanced-regression-techniques/train.csv'
    path_test = 'house-prices-advanced-regression-techniques/test.csv'
    root_pth = 'weight'
    root_pred = 'result'
    os.makedirs(root_pth, exist_ok=True)
    os.makedirs(root_pred, exist_ok=True)
    k = 10
    batch_size = 64
    num_epoches = 100
    lr = 0.001
    weight_decay = 0.01
    num_hidden = 128

    np_train, np_price, np_test = dataset_process(path_train, path_test)
    traindataset = HousePrice_TrainDataset(path_train, path_test)
    train_datasetloader = HousePrice_Datasetloader(traindataset, k, batch_size)
    num_features = np_train.shape[-1]

    # baseline_net = Baseline_Net(num_features)
    # baseline_writer = SummaryWriter(f'runs/baseline_net_{lr}_{num_epoches}')
    # Optim = Optimizer(baseline_net.parameters(), lr, weight_decay)
    # train(train_datasetloader, baseline_net, loss_fn, Optim, num_epoches, baseline_writer)
    # baseline_writer.close()
    # torch.save(baseline_net.state_dict(), f"{root_pred}/baseline_net_{lr}_{num_epoches}.pth")

    name_file = f'mlp_net_{num_hidden}_{lr}_{num_epoches}_{batch_size}_{weight_decay}'
    mlp = MLP_Net(num_features, num_hidden)
    mlp_writer = SummaryWriter(f'runs/{name_file}')
    Optim = Optimizer(mlp.parameters(), lr, weight_decay)
    train(train_datasetloader, mlp, loss_fn, Optim, num_epoches, mlp_writer)
    mlp_writer.close()
    save_pth = f"{root_pth}/{name_file}.pth"
    torch.save(mlp.state_dict(), save_pth)
    pred(mlp, save_pth, np_test)
