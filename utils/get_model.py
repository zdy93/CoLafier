# -*- coding: utf-8 -*-
# @Author : Jack (this code is copied from Cheng Tan's codebase "Co-training-based_noisy-label-learning-master". Thank you!)
# @Email  : liyifan20g@ict.ac.cn
# @File   : get model.py


import models
from torchvision.models import resnet50, vgg19_bn
import torch.nn as nn

model_names = sorted(name for name in models.__dict__
                     if callable(models.__dict__[name]))


def get_model(
    model_name: str = 'resnet18',
    input_channel: int = 3,
    num_classes: int = 10,
    device=None,
    with_label: bool = False,
):

    assert (model_name in model_names) or (model_name == 'VGG-19')
    if model_name.startswith('resnet'):
        if model_name.startswith('resnet50'):
            if with_label:
                model_name = 'resnet50withlabel'
                model = models.__dict__[model_name](pretrained=True)
                model.label_embedding = nn.Embedding(num_classes, 2048)
                model.layerNorm = nn.LayerNorm(2048)
                model.fc = nn.Linear(2048, num_classes)
            else:
                model = resnet50(pretrained=True)
                model.fc = nn.Linear(2048, num_classes)
        elif with_label:
            model = models.__dict__[model_name](input_channel=input_channel,
                                                num_classes=num_classes,
                                                with_label=with_label)
        else:
            model = models.__dict__[model_name](input_channel=input_channel,
                                                num_classes=num_classes)
    elif model_name.startswith('PreResNet'):
        if with_label:
            model = models.__dict__[model_name](num_classes=num_classes,
                                                with_label=with_label)
        else:
            model = models.__dict__[model_name](num_classes=num_classes)
    elif model_name.startswith('VGG'):
        if with_label:
            model_name = 'vgg19_bn_with_label'
            model = models.__dict__[model_name](pretrained=False, num_classes=num_classes)
        else:
            model = vgg19_bn(pretrained=False)
            model.classifier._modules['6'] = nn.Linear(4096, num_classes)
    elif model_name.startswith('Linear'):
        model = models.__dict__[model_name](num_classes=num_classes,
                                            with_label=with_label)

    return model.to(device)
