import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import numpy as np
from utils import get_model, get_performance
from sklearn.mixture import GaussianMixture

class DivideMixTrain:

    def __init__(self, config: dict = None, input_channel: int = 3, num_classes: int = 10,
                 noise_mode='sym', alpha=4, lambda_u=25, p_threshold=0.5, T=0.5, num_epochs=300, r=0.5, id='', seed=123, gpuid=0, num_class=10, data_path='./cifar-10', dataset='cifar10'):
        self.lr = config['lr']
        self.noise_type = config['noise_type']
        self.alpha = config['alpha']
        self.lambda_u = config['lambda_u']
        self.p_threshold = config['p_threshold']
        self.T = config['T']
        self.num_epochs = config['epochs']
        self.r = config['r']
        self.id = config['id']
        self.seed = config['seed']
        device = torch.device('cuda:%s'%config['gpu']) if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.num_class = num_classes

        torch.cuda.set_device(self.gpuid)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.net1 = get_model(config['model1_type'], input_channel, num_classes, device)
        self.net2 = get_model(config['model2_type'], input_channel, num_classes, device)
        self.criterion = self.SemiLoss()
        self.optimizer1 = optim.SGD(self.net1.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.optimizer2 = optim.SGD(self.net2.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.CE = nn.CrossEntropyLoss(reduction='none')
        self.CEloss = nn.CrossEntropyLoss()
        if self.noise_type == 'asym':
            self.conf_penalty = self.NegEntropy()
        self.all_loss = [[], []]

    def train(self, epoch):
        pass
    # ... contents of the train function

    def warmup(self, epoch):
        pass
    # ... contents of the warmup function

    def test(self, epoch):
        pass
    # ... contents of the test function

    def eval_train(self):
        pass
    # ... contents of the eval_train function

    def linear_rampup(self, current, warm_up, rampup_length=16):
        pass
    # ... contents of the linear_rampup function

    class SemiLoss:
        pass
    # ... contents of the SemiLoss class

    class NegEntropy:
        pass
    # ... contents of the NegEntropy class

# ... other utility functions and classes
