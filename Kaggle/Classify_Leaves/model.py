# -*- coding: utf-8 -*-
# @Author  : Zhangwei
# @Time    : 2024/12/17 下午8:35
import numpy as np
import torch
import torchvision.models as models
from torch import nn
from torchvision.models import ResNet34_Weights
from torchsummary import summary


# 是否要冻住模型的前面一些层
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def res_model(num_classes: int, feature_extract=False, use_pretrained=True):
    if use_pretrained:
        weights = ResNet34_Weights.IMAGENET1K_V1  # 或 ResNet34_Weights.DEFAULT
    else:
        weights = None

    model_ft = models.resnet34(weights=weights)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    nn.init.xavier_uniform_(model_ft.fc.weight)
    return model_ft


if __name__ == '__main__':
    model = res_model(10)
    for name, module in model.named_parameters():
        print(name, module)
