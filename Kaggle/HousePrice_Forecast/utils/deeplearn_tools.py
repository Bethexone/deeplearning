import torch
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        backend_inline.set_matplotlib_formats('svg')
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda:set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for X, Y, fmt in zip(self.X, self.Y, self.fmts):
            X = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in X ]
            Y = [y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y for y in Y ]
            self.axes[0].plot(X, Y, fmt)
        self.config_axes()

        display.display(self.fig)
        display.clear_output(wait=True)
        # Show the plot (this is the part that was changed)
        # plt.draw()  # Redraw the current figure
        # plt.pause(0.01)  # Pause to allow the plot to update

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴
    Defined in :numref:`sec_calculus`"""
  
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def image_show(images, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images"""
    figsize = (num_cols * scale, num_rows * scale)
    figs, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
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

def synthetic_data(true_w,true_b,num_data):
    mean,std = 0.0,1.0
    x = torch.normal(mean,std,size=(num_data,len(true_w)))
    y = torch.matmul(x,true_w)+true_b
    y += torch.normal(0,0.01,y.shape)
    return  x,y

def evaluate_accuracy(net,data_iter,device='cpu'):
    """计算在指定数据集的精度"""
    if isinstance(net,torch.nn.Module):
        # 评估模式，不会计算梯度
        net.eval()
        device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    for X,y in data_iter:
      if isinstance(X,list):
        X = [x.to(device) for x in X]
      else:
        X = X.to(device)
      y = y.to(device)
      metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]

def accuracy(y_hat,y):
    y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y.reshape(y_hat.shape)
    return float(cmp.sum())

class Timer:
    """记录多次运行时间"""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()
