# -*- coding: utf-8 -*-
# @Author : Jack (this code is copied from Cheng Tan's codebase "Co-training-based_noisy-label-learning-master". Thank you!)
# @Email  : liyifan20g@ict.ac.cn
# @File   : Coteaching.py (refer to Bo Han's NeurlPS 2020 paper "Robust Training of Deep Neural Networks with Extremely Noisy Labels")

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from utils import get_model, get_performance
from losses import loss_coteaching
from tqdm import tqdm
import time


class Coteaching:
    def __init__(
            self, 
            config: dict = None, 
            input_channel: int = 3, 
            num_classes: int = 10,
        ):

        self.lr = config['lr']

        if config['forget_rate'] is None:
            if config['noise_type'] == 'asym':
                forget_rate = config['percent'] / 2
            else:
                forget_rate = config['percent']
        else:
            forget_rate = config['forget_rate']

        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        self.alpha_plan = [self.lr] * config['epochs']
        self.beta1_plan = [mom1] * config['epochs']

        for i in range(config['epoch_decay_start'], config['epochs']):
            self.alpha_plan[i] = float(config['epochs'] - i) / (config['epochs'] - config['epoch_decay_start']) * self.lr
            self.beta1_plan[i] = mom2
            
        device = torch.device('cuda:%s'%config['gpu']) if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        config['device'] = torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'cpu'
        self.epochs = config['epochs']

        # define drop rate schedule
        self.rate_schedule = np.ones(config['epochs']) * forget_rate
        self.rate_schedule[:config['num_gradual']] = np.linspace(0, forget_rate ** config['exponent'], config['num_gradual'])

        # model
        self.model1 = get_model(config['model1_type'], input_channel, num_classes, device)
        self.model2 = get_model(config['model2_type'], input_channel, num_classes, device)

        self.adjust_lr = config['adjust_lr']
        assert config['optimizer'] in  ['adam', 'sgd']
        if config['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()), lr=self.lr)
        else:
            self.weight_decay = config['weight_decay']
            self.optimizer = torch.optim.SGD(list(self.model1.parameters()) + list(self.model2.parameters()),
                                             lr=self.lr, weight_decay=self.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config['milestones'], verbose=True)
        self.optim_type = config['optimizer']
        self.loss_fn = loss_coteaching

    def evaluate(self, test_loader):
        print('Evaluating ...')
        self.model1.eval()  # Change model to 'eval' mode.
        self.model2.eval()  # Change model to 'eval' mode

        correct1 = 0
        total1 = 0
        correct2 = 0
        total2 = 0
        target_list, pred_1_list, pred_2_list = [], [], []
        eval_time = time.time()
        with torch.no_grad():
            for images, labels in test_loader:
                target_list.append(labels.cpu().numpy())
                images = Variable(images).to(self.device)
                logits1 = self.model1(images)
                outputs1 = F.softmax(logits1, dim=1)
                _, pred1 = torch.max(outputs1.data, 1)

                pred_1_list.append(pred1.cpu().numpy())
                total1 += labels.size(0)
                correct1 += (pred1.cpu() == labels).sum()

                logits2 = self.model2(images)
                outputs2 = F.softmax(logits2, dim=1)
                _, pred2 = torch.max(outputs2.data, 1)

                pred_2_list.append(pred2.cpu().numpy())
                total2 += labels.size(0)
                correct2 += (pred2.cpu() == labels).sum()
        target = np.concatenate(target_list)
        pred_1 = np.concatenate(pred_1_list)
        pred_2 = np.concatenate(pred_2_list)
        eval_dict_1, eval_dict_2 = get_performance(target, pred_1), get_performance(target, pred_2)
        eval_dict_1['eval_time'], eval_dict_2['eval_time'] = eval_time, eval_time
        acc1 = 100 * float(correct1) / float(total1)
        acc2 = 100 * float(correct2) / float(total2)
        return eval_dict_1, eval_dict_2

    def train(self, train_loader, epoch):
        print('Training ...')
        self.model1.train()  
        self.model2.train()


        pbar = tqdm(train_loader)
        for (images, labels) in pbar:

            images = Variable(images).to(self.device)
            labels = Variable(labels).to(self.device)

            logits1 = self.model1(images)
            logits2 = self.model2(images)

            loss_1, loss_2 = self.loss_fn(logits1, logits2, labels, self.rate_schedule[epoch])

            self.optimizer.zero_grad()
            loss_1.backward()
            loss_2.backward()
            self.optimizer.step()

            pbar.set_description(
                    'Epoch [%d/%d], Loss1: %.4f, Loss2: %.4f'
                    % (epoch + 1, self.epochs, loss_1.data.item(), loss_2.data.item()))

        if self.adjust_lr == 1:
            if self.optim_type == 'adam':
                self.adjust_learning_rate(self.optimizer, epoch)
            elif self.optim_type == 'sgd':
                self.scheduler.step()

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1