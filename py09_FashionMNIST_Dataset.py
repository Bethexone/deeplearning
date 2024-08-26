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
