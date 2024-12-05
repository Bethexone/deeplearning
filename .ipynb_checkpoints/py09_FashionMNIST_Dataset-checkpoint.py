import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from matplotlib import pyplot as plt

from matplotlib_inline import backend_inline

backend_inline.set_matplotlib_formats('svg')

trans = transforms.ToTensor()

mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)


def get_dataloader_worker():
    return 4


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST 数据集的文本标签"""
    text_labels = mnist_train.classes
    return [text_labels[int(i)] for i in labels]


def image_show(images, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images"""
    figsize = (num_cols * scale, num_rows * scale)
    fi, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, images)):
        if torch.is_tensor(img):
            # 图片张量s
            ax.imshow(img.numpy())
        else:
            # PIL 图片
            ax.imshow(img)
        if titles:
            ax.set_title(titles[i])
        ax.axis('off')
    plt.subplots_adjust(wspace=0.3, hspace=0.1)
    return axes


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
    return data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_worker()), \
        data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=get_dataloader_worker())


def train(train_loader,test_loader,net,num_epoches,loss_fn,optimizer,device="cuda"):
    def init_weights(layer):
        if type(layer)== nn.Linear or type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)
    net.apply(init_weights)
    net.to(device)
    data_dis = Animator(xlabel='num_epoch',xlim=[1,num_epoches],legend=['train_loss','train_accuracy','test_accuracy'],yscale='log')
    timer = Timer()
    for epoch in range(num_epoches):
        l_sum = 0
        timer.start()
        for X,y in train_loader:
            X,y = X.to(device),y.to(device)
            optimizer.zero_grad()
            y_hat = alexnet(X)
            l = loss_fn(y_hat,y)
            l.backward()
            optimizer.step()
            l_sum += l
        epoch_time = timer.stop()
        train_acc = evaluate_accuracy(net,train_loader,device)
        test_acc = evaluate_accuracy(net,test_loader,device)
        print(f"{epoch}:train loss,{l_sum:.2f},train_acc:{train_acc},test_acc:{test_acc},time:{epoch_time:.4f}s")
        data_dis.add(epoch+1,[l_sum,train_acc,test_acc])
