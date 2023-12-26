import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
from utils import get_model, get_performance, FMix
from losses import loss_lid_general, GCELoss, loss_consistency, loss_lid, loss_lid_cos
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.distributions.beta import Beta
import time
import wandb


def record_history(history_dict, key_list, value_tensor_list, epoch, indexes):
    for key, value_tensor in zip(key_list, value_tensor_list):
        if value_tensor is not None:
            history_dict[key][indexes, epoch] = value_tensor.cpu()


def lid(batch, k, distmetric='l2') -> torch.Tensor:
    k = min(k, batch.shape[0] - 1)
    if distmetric == 'cos':
        a_norm = batch / batch.norm(p=2, dim=1)[:, None]

        # cosine distance: 1-cos()
        cos_sim = torch.mm(a_norm, a_norm.transpose(0, 1))
        cos_distance = torch.ones(cos_sim.size()).to(batch.device) - cos_sim
        distance_sorted, indices = torch.sort(cos_distance)
    else:
        assert distmetric == 'l2'
        distance = torch.cdist(batch, batch, p=2)
        distance_sorted, indices = torch.sort(distance)

    selected = distance_sorted[:, 1:k + 1] + 1e-12
    lids_log_term = torch.sum(torch.log(selected / (selected[:, -1]).reshape(-1, 1)), dim=1)
    lids_log_term = lids_log_term + 1e-12
    lids = -k / lids_log_term
    return lids


class CoLafier:

    def __init__(
            self,
            config: dict = None,
            input_channel: int = 3,
            num_classes: int = 10,
    ):

        self.set_model_and_learner(config, input_channel, num_classes)
        self.input_channel = input_channel
        self.num_classes = num_classes

        # LID
        self.lid_k = config['lid_k']
        self.lid_metric = config['lid_metric']

        dataset_sizes = {
            'cifar-10': 50000,
            'cifar-10N': 50000,
        }
        self.N = dataset_sizes[config['dataset']]
        self.labels = -1 * torch.ones(self.N, dtype=torch.int64, device=self.device)
        self.label_update_record = self.labels.detach().clone()
        self.last_labels = self.labels.detach().clone()
        self.ground_truth_label = None
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(config['seed'])
        self.num_classes = num_classes
        self.input_channel = input_channel
        self.training_config = config
        self.consider_diff = config['consider_diff']
        self.noise_ratio = config['percent']
        self.alpha = config['alpha']
        self.mixup_loss = config['mixup_loss']
        self.label_update_ratio = None

        self.set_schedule(config)

        self.record_history = config['record_history']
        self.history_lids = torch.zeros((self.N, self.epochs))
        self.mix_type = config['mix_type']
        self.GCE_loss = GCELoss(num_classes=num_classes, gpu=config['gpu'], reduction='none')
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.merge_concat = config['merge_concat']
        self.activation = config['activation']
        self.history_weak_lids = torch.zeros((self.N, self.epochs), device=self.device)
        self.history_strong_lids = torch.zeros((self.N, self.epochs), device=self.device)
        self.history_labels = torch.zeros((self.N, self.epochs), dtype=torch.int64, device=self.device)
        self.lambda_clean, self.lambda_hard, self.lambda_mix = config['lambda_clean'], config['lambda_hard'], config[
            'lambda_mix']
        self.new_label_loss = config['new_label_loss']
        self.lambda_new = config['lambda_new']
        self.consistency_loss = config['consistency_loss']
        self.lambda_cons = config['lambda_cons']
        self.history_loss_1 = {i: torch.zeros((self.N, self.epochs)) for i in ['clean', 'hard', 'mix']}
        self.history_loss_2 = {i: torch.zeros((self.N, self.epochs)) for i in
                               ['clean', 'hard', 'mix', 'new_clean', 'new_hard', 'cons']}
        self.history_weight = {i: torch.zeros((self.N, self.epochs)) for i in
                               ['clean', 'hard', 'noisy', 'w_pseudo', 's_pseudo']}
        self.embedding_type = config['embedding_type']
        self.u_lid = config['u_lid']

    def set_model_and_learner(self, config, input_channel, num_classes):
        self.lr = config['lr']

        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        self.alpha_plan = [self.lr] * config['epochs']
        self.beta1_plan = [mom1] * config['epochs']
        self.beta2 = config['beta2']

        for i in range(config['epoch_decay_start'], config['epochs']):
            self.alpha_plan[i] = float(config['epochs'] - i) / (
                    config['epochs'] - config['epoch_decay_start']) * self.lr
            self.beta1_plan[i] = mom2

        device = torch.device('cuda:%s' % config['gpu']) if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        config['device'] = torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'cpu'
        self.epochs = config['epochs']

        # model
        self.model1 = get_model(config['model1_type'], input_channel, num_classes, device)
        self.model2 = get_model(config['model2_type'], input_channel, num_classes, device, True)

        assert config['optimizer'] in ['adam', 'adamw', 'sgd']
        self.adjust_lr = config['adjust_lr']
        self.no_decay = config['no_decay']
        self.weight_decay = config['weight_decay']
        if self.no_decay:
            param_optimizer = list(self.model1.named_parameters()) + list(self.model2.named_parameters())
            no_decay = ['bias', "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': self.weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        if config['optimizer'] == 'adam':
            if self.no_decay:
                self.optimizer = torch.optim.Adam(optimizer_grouped_parameters,
                                                  lr=self.lr, betas=(self.beta1_plan[0], self.beta2))
            else:
                self.optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()),
                                                  lr=self.lr, weight_decay=self.weight_decay,
                                                  betas=(self.beta1_plan[0], self.beta2))
        elif config['optimizer'] == 'adamw':
            if self.no_decay:
                self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                                   lr=self.lr, betas=(self.beta1_plan[0], self.beta2))
            else:
                self.optimizer = torch.optim.AdamW(list(self.model1.parameters()) + list(self.model2.parameters()),
                                                   lr=self.lr, weight_decay=self.weight_decay,
                                                   betas=(self.beta1_plan[0], self.beta2))
        else:
            if self.no_decay:
                self.optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=self.lr)
            else:
                self.optimizer = torch.optim.SGD(list(self.model1.parameters()) + list(self.model2.parameters()),
                                                 lr=self.lr, weight_decay=self.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config['milestones'],
                                                                  verbose=True)
        self.optim_type = config['optimizer']
        if config['consistency_loss']:
            self.loss_fn = loss_lid_cos
        else:
            self.loss_fn = loss_lid

    def set_schedule(self, config):
        self.current_epoch = -1
        self.warmup_epoch = config['warmup_epoch']
        self.keep_threshold_schedule = np.ones(config['epochs']) * config['keep_threshold_end']
        if config['num_keep_gradual'] > 0:
            num_keep_gradual = min(config['num_keep_gradual'], self.epochs - self.warmup_epoch)
            self.keep_threshold_schedule[:self.warmup_epoch] = np.ones(self.warmup_epoch) * config[
                'keep_threshold_init']
            self.keep_threshold_schedule[self.warmup_epoch:num_keep_gradual + self.warmup_epoch] = \
                np.linspace(config['keep_threshold_init'], config['keep_threshold_end'], num_keep_gradual)

        self.low_quant_schedule = np.ones(config['epochs']) * config['low_quant_end']
        self.high_quant_schedule = np.ones(config['epochs']) * config['high_quant_end']
        num_quant_gradual = min(config['num_quant_gradual'], self.epochs - self.warmup_epoch)
        if config['num_quant_gradual'] > 0:
            self.low_quant_schedule[:self.warmup_epoch] = np.ones(self.warmup_epoch) * config['low_quant_init']
            self.high_quant_schedule[:self.warmup_epoch] = np.ones(self.warmup_epoch) * config['high_quant_init']
            self.low_quant_schedule[self.warmup_epoch: num_quant_gradual + self.warmup_epoch] = \
                np.linspace(config['low_quant_init'], config['low_quant_end'], num_quant_gradual)
            self.high_quant_schedule[self.warmup_epoch: num_quant_gradual + self.warmup_epoch] = \
                np.linspace(config['high_quant_init'], config['high_quant_end'], num_quant_gradual)

        self.loss_low_quant_schedule = np.ones(config['epochs']) * config['loss_low_quant_end']
        self.loss_high_quant_schedule = np.ones(config['epochs']) * config['loss_high_quant_end']
        num_loss_quant_gradual = min(config['num_loss_quant_gradual'], self.epochs - self.warmup_epoch)
        if config['num_loss_quant_gradual'] > 0:
            self.loss_low_quant_schedule[:self.warmup_epoch] = np.ones(self.warmup_epoch) * config[
                'loss_low_quant_init']
            self.loss_high_quant_schedule[:self.warmup_epoch] = np.ones(self.warmup_epoch) * config[
                'loss_high_quant_init']
            self.loss_low_quant_schedule[self.warmup_epoch: num_loss_quant_gradual + self.warmup_epoch] = \
                np.linspace(config['loss_low_quant_init'], config['loss_low_quant_end'], num_loss_quant_gradual)
            self.loss_high_quant_schedule[self.warmup_epoch: num_loss_quant_gradual + self.warmup_epoch] = \
                np.linspace(config['loss_high_quant_init'], config['loss_high_quant_end'], num_loss_quant_gradual)

        self.hist_w_low_quant, self.hist_w_high_quant, self.hist_s_low_quant, self.hist_s_high_quant = None, None, None, None

    def get_labels(self, train_loader):
        print("Loading labels......")
        for (_, labels, indexes) in train_loader:
            labels = labels.to(self.device)
            self.labels[indexes] = labels
        self.ground_truth_label = train_loader.dataset.ground_truth_targets
        self.last_labels = self.labels.detach().clone()
        print("The labels are loaded!")

    def get_pseudo_label(self, indexes):
        pseudo_label = self.labels[indexes]
        return pseudo_label

    def assign_new_label(self, labels):
        label_weights = -1.0 * F.one_hot(labels, num_classes=self.num_classes)
        label_weights[label_weights == 0] = 1.0
        label_weights[label_weights == -1] = 0.0
        new_label = torch.multinomial(label_weights, 1, generator=self.rng, replacement=True).squeeze()
        return new_label

    def get_loss_weight(self, lids, epoch):
        if epoch < self.warmup_epoch:
            w_lids = lids[0]
            clean_weight, noisy_weight, hard_weight = torch.ones_like(w_lids), \
                torch.zeros_like(w_lids), torch.zeros_like(w_lids)
        else:
            low_quant_thres = self.loss_low_quant_schedule[epoch]
            high_quant_thres = self.loss_high_quant_schedule[epoch]
            w_lids, s_lids = lids
            w_low_bound = torch.quantile(w_lids, low_quant_thres)
            w_high_bound = torch.quantile(w_lids, high_quant_thres)
            s_low_bound = torch.quantile(s_lids, low_quant_thres)
            s_high_bound = torch.quantile(s_lids, high_quant_thres)
            w_weight = torch.clamp((w_high_bound - w_lids) / (w_high_bound - w_low_bound), min=0.,
                                   max=1.)
            s_weight = torch.clamp((s_high_bound - s_lids) / (s_high_bound - s_low_bound), min=0.,
                                   max=1.)
            w_s_weight = torch.stack([w_weight, s_weight], dim=1)
            clean_weight = torch.min(w_s_weight, dim=1).values
            noisy_weight = torch.min(1.0 - w_s_weight, dim=1).values
            hard_weight = torch.abs((w_weight - s_weight))
        return clean_weight, noisy_weight, hard_weight

    def update_pseudo_label_ws(self, last_labels, new_pseudo_labels, w_weight, s_weight, indexes, epoch):
        if epoch < self.warmup_epoch:
            pass
        else:
            keep_threshold = self.keep_threshold_schedule[epoch]
            w_new_pseudo_labels = new_pseudo_labels[:new_pseudo_labels.shape[0] // 2, ]
            s_new_pseudo_labels = new_pseudo_labels[new_pseudo_labels.shape[0] // 2:, ]
            update_cond = (w_weight > keep_threshold) & (s_weight > keep_threshold) & (
                    w_new_pseudo_labels == s_new_pseudo_labels)
            self.labels[indexes] = torch.where(update_cond, w_new_pseudo_labels, last_labels)
            self.label_update_record[indexes] = torch.where(update_cond, w_new_pseudo_labels,
                                                            -1 * torch.ones_like(w_new_pseudo_labels))


    def get_two_pseudo_weight(self, w_lids, s_lids, ori_pseudo_pred_diff, new_pseudo_pred_diff, epoch):
        if epoch < self.warmup_epoch:
            w_weight_pseudo = torch.zeros_like(w_lids[:w_lids.shape[0] // 2, ])
            s_weight_pseudo = torch.zeros_like(s_lids[:s_lids.shape[0] // 2, ])
        else:
            w_ori_lids = w_lids[:w_lids.shape[0] // 2, ]
            w_pseudo_lids = w_lids[w_lids.shape[0] // 2:, ]
            s_ori_lids = s_lids[:s_lids.shape[0] // 2, ]
            s_pseudo_lids = s_lids[s_lids.shape[0] // 2:, ]
            w_ori_pseudo_pred_diff = ori_pseudo_pred_diff[:ori_pseudo_pred_diff.shape[0] // 2, ]
            s_ori_pseudo_pred_diff = ori_pseudo_pred_diff[ori_pseudo_pred_diff.shape[0] // 2:, ]
            w_new_pseudo_pred_diff = new_pseudo_pred_diff[:new_pseudo_pred_diff.shape[0] // 2, ]
            s_new_pseudo_pred_diff = new_pseudo_pred_diff[new_pseudo_pred_diff.shape[0] // 2:, ]
            low_quant_thres = self.low_quant_schedule[epoch]
            high_quant_thres = self.high_quant_schedule[epoch]
            w_low_quant = torch.quantile(w_lids, low_quant_thres)
            w_high_quant = torch.quantile(w_lids, high_quant_thres)
            s_low_quant = torch.quantile(s_lids, low_quant_thres)
            s_high_quant = torch.quantile(s_lids, high_quant_thres)
            if self.consider_diff:
                w_weight_ori = torch.clamp(
                    ((2 - w_ori_pseudo_pred_diff) / 2) * (
                            (w_high_quant - w_ori_lids) / (w_high_quant - w_low_quant)),
                    min=0., max=1.)
                s_weight_ori = torch.clamp(
                    ((2 - s_ori_pseudo_pred_diff) / 2) * (
                            (s_high_quant - s_ori_lids) / (s_high_quant - s_low_quant)),
                    min=0., max=1.)
                w_weight_pseudo = torch.clamp(
                    ((2 - w_new_pseudo_pred_diff) / 2) * (
                            (w_high_quant - w_pseudo_lids) / (w_high_quant - w_low_quant)),
                    min=0., max=1.)
                s_weight_pseudo = torch.clamp(
                    ((2 - s_new_pseudo_pred_diff) / 2) * (
                            (s_high_quant - s_pseudo_lids) / (s_high_quant - s_low_quant)),
                    min=0., max=1.)
            else:
                w_weight_ori = torch.clamp((w_high_quant - w_ori_lids) / (w_high_quant - w_low_quant), min=0.,
                                           max=1.)
                w_weight_pseudo = torch.clamp((w_high_quant - w_pseudo_lids) / (w_high_quant - w_low_quant), min=0.,
                                              max=1.)
                s_weight_ori = torch.clamp((s_high_quant - s_ori_lids) / (s_high_quant - s_low_quant), min=0.,
                                           max=1.)
                s_weight_pseudo = torch.clamp((s_high_quant - s_pseudo_lids) / (s_high_quant - s_low_quant), min=0.,
                                              max=1.)
            w_weight_pseudo = torch.clamp(torch.sign(w_weight_pseudo - w_weight_ori) * w_weight_pseudo, min=0.,
                                          max=1.)
            s_weight_pseudo = torch.clamp(torch.sign(s_weight_pseudo - s_weight_ori) * s_weight_pseudo, min=0.,
                                          max=1.)
        return w_weight_pseudo, s_weight_pseudo

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
                labels_gpu = Variable(labels).to(self.device)
                logits1 = self.model1(images)
                outputs1 = F.softmax(logits1, dim=1)
                _, pred1 = torch.max(outputs1.data, 1)

                pred_1_list.append(pred1.cpu().numpy())
                total1 += labels.size(0)
                correct1 += (pred1.cpu() == labels).sum()

                logits2 = self.model2(images, labels_gpu, get_feat=False,
                                      merge_concat=self.merge_concat, activation=self.activation,
                                      embedding_type=self.embedding_type)
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
        for k, v in eval_dict_2.items():
            eval_dict_1[f'm2_{k}'] = v
        if self.ground_truth_label is not None:
            pseudo_acc = accuracy_score(self.ground_truth_label, self.labels.cpu().numpy())
        else:
            print('Ground Truth Label is None')
            pseudo_acc = 1 - self.noise_ratio
        eval_dict_1['pseudo_acc'] = pseudo_acc
        self.label_update_ratio = 1 - accuracy_score(self.last_labels.cpu().numpy(), self.labels.cpu().numpy())
        eval_dict_1['label_update_ratio'] = self.label_update_ratio
        log_dict = {key + "_model_1": value for key, value in eval_dict_1.items() if not key.startswith("m2_")}
        log_dict.update({key + "_model_2": value for key, value in eval_dict_2.items()})
        log_dict['epoch'] = self.current_epoch
        wandb.log(log_dict)
        return eval_dict_1  # keep model 1 performance only

    def train(self, train_loader, epoch):
        print('Training ...')
        self.current_epoch = epoch
        self.model1.train()
        self.model2.train()

        self.last_labels = self.labels.detach().clone()
        pbar = tqdm(train_loader)
        for (two_images, labels, indexes) in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            b = len(labels)
            w_imgs, s_imgs = (two_images[0].to(self.device, non_blocking=True),
                              two_images[1].to(self.device, non_blocking=True))
            labels = labels.to(self.device)
            pseudo_labels = self.get_pseudo_label(indexes)
            new_labels = self.assign_new_label(pseudo_labels)

            images = torch.cat([w_imgs, s_imgs], dim=0)
            double_pseudo_labels = torch.cat([pseudo_labels, pseudo_labels], dim=0)
            double_new_labels = torch.cat([new_labels, new_labels], dim=0)
            bs = w_imgs.shape[0]
            logits1 = self.model1(images)
            logits2, feature2 = self.model2(images, double_pseudo_labels, merge_concat=self.merge_concat,
                                            activation=self.activation, embedding_type=self.embedding_type)
            logits2_new, feature2_new = self.model2(images, double_new_labels, merge_concat=self.merge_concat,
                                                    activation=self.activation, embedding_type=self.embedding_type)
            outputs1 = F.softmax(logits1, dim=1)
            outputs2 = F.softmax(logits2, dim=1)
            pred1 = torch.argmax(outputs1.data, 1).detach()
            logits2_pseudo, feature2_pseudo = self.model2(images, pred1, merge_concat=self.merge_concat,
                                                          activation=self.activation,
                                                          embedding_type=self.embedding_type)
            outputs2_pseudo = F.softmax(logits2_pseudo, dim=1)
            onehot_labels = torch.zeros(b, self.num_classes, device=self.device).scatter_(1, pseudo_labels.view(-1, 1), 1)
            w_mixed_images, w_mixed_labels, w_lam, w_second_label_indexes = self.org_mixup_data(w_imgs, onehot_labels)
            s_mixed_images, s_mixed_labels, s_lam, s_second_label_indexes = self.org_mixup_data(s_imgs, onehot_labels)
            w_logits1_mix = self.model1(w_mixed_images)
            s_logits1_mix = self.model1(s_mixed_images)
            w_logits2_mix, w_feature2_mix = self.model2(w_mixed_images, pseudo_labels,
                                                        merge_concat=self.merge_concat, activation=self.activation,
                                                        label_2=pseudo_labels[w_second_label_indexes], lam=w_lam,
                                                        embedding_type=self.embedding_type)
            s_logits2_mix, s_feature2_mix = self.model2(s_mixed_images, pseudo_labels,
                                                        merge_concat=self.merge_concat, activation=self.activation,
                                                        label_2=pseudo_labels[s_second_label_indexes], lam=s_lam,
                                                        embedding_type=self.embedding_type)

            w_loss_1_mix = -torch.sum(F.log_softmax(w_logits1_mix, dim=1) * w_mixed_labels, dim=1)
            s_loss_1_mix = -torch.sum(F.log_softmax(s_logits1_mix, dim=1) * s_mixed_labels, dim=1)
            w_loss_2_mix = -torch.sum(F.log_softmax(w_logits2_mix, dim=1) * w_mixed_labels, dim=1)
            s_loss_2_mix = -torch.sum(F.log_softmax(s_logits2_mix, dim=1) * s_mixed_labels, dim=1)

            with torch.no_grad():
                ori_pseudo_outputs_diff = torch.abs(outputs1 - outputs2).sum(axis=1)
                new_pseudo_outputs_pseudo_diff = torch.abs(outputs1 - outputs2_pseudo).sum(axis=1)
                w_feature2_ori_pseudo = torch.cat((feature2[:bs, ], feature2_pseudo[:bs, ]), 0)
                s_feature2_ori_pseudo = torch.cat((feature2[bs:, ], feature2_pseudo[bs:, ]), 0)
                w_lid_score_ori_pseudo = lid(w_feature2_ori_pseudo, self.lid_k, self.lid_metric)
                s_lid_score_ori_pseudo = lid(s_feature2_ori_pseudo, self.lid_k, self.lid_metric)
                if not self.u_lid:
                    lid_score_2 = lid(feature2, self.lid_k, self.lid_metric)
                    clean_weight_loss, noisy_weight_loss, hard_weight_loss = self.get_loss_weight(
                        (lid_score_2[:bs], lid_score_2[bs:]), epoch)
                else:
                    clean_weight_loss, noisy_weight_loss, hard_weight_loss = self.get_loss_weight(
                        (w_lid_score_ori_pseudo[:bs], s_lid_score_ori_pseudo[:bs]), epoch)
                w_weight_pseudo, s_weight_pseudo = self.get_two_pseudo_weight(w_lid_score_ori_pseudo,
                                                                              s_lid_score_ori_pseudo,
                                                                              ori_pseudo_outputs_diff,
                                                                              new_pseudo_outputs_pseudo_diff, epoch)

            loss_1_clean, loss_2_clean, loss_2_new_clean = loss_lid_general(logits1, logits2, logits2_new,
                                                                            pseudo_labels, clean_weight_loss,
                                                                            self.criterion)
            loss_1_hard, loss_2_hard, loss_2_new_hard = loss_lid_general(logits1, logits2, logits2_new, pseudo_labels,
                                                                         hard_weight_loss, self.GCE_loss)
            loss_1_mix = (noisy_weight_loss * w_loss_1_mix).mean() + (noisy_weight_loss * s_loss_1_mix).mean()
            loss_2_mix = (noisy_weight_loss * w_loss_2_mix).mean() + (noisy_weight_loss * s_loss_2_mix).mean()
            loss_1 = (
                         self.lambda_clean if epoch >= self.warmup_epoch else 1.0) * loss_1_clean + self.lambda_hard * loss_1_hard + self.lambda_mix * loss_1_mix
            if self.new_label_loss:
                loss_2 = (self.lambda_clean if epoch >= self.warmup_epoch else 1.0) * loss_2_clean + (
                    self.lambda_clean if epoch >= self.warmup_epoch else 1.0) * self.lambda_new * loss_2_new_clean + self.lambda_hard * loss_2_hard + self.lambda_hard * self.lambda_new * loss_2_new_hard + self.lambda_mix * loss_2_mix
            else:
                loss_2 = (
                             self.lambda_clean if epoch >= self.warmup_epoch else 1.0) * loss_2_clean + self.lambda_hard * loss_2_hard + self.lambda_mix * loss_2_mix
            loss_2_cons = loss_consistency(logits2, logits2_new, noisy_weight_loss)
            if self.consistency_loss and epoch >= self.warmup_epoch:
                loss_2 = loss_2 + self.lambda_cons * loss_2_cons
            with torch.no_grad():
                self.update_pseudo_label_ws(pseudo_labels, pred1, w_weight_pseudo, s_weight_pseudo, indexes, epoch)
                if not self.u_lid:
                    self.history_weak_lids[indexes, epoch] = lid_score_2[:b]
                    self.history_strong_lids[indexes, epoch] = lid_score_2[b:]
                else:
                    self.history_weak_lids[indexes, epoch] = w_lid_score_ori_pseudo[:b]
                    self.history_strong_lids[indexes, epoch] = s_lid_score_ori_pseudo[:b]

                self.history_labels[indexes, epoch] = self.labels[indexes]
                loss_1_key_list = ['clean', 'hard', 'mix']
                loss_1_value_list = [loss_1_clean, loss_1_hard, loss_1_mix]
                loss_2_key_list = ['clean', 'hard', 'mix', 'new_clean', 'new_hard', 'cons']
                loss_2_value_list = [loss_2_clean, loss_2_hard, loss_2_mix, loss_2_new_clean, loss_2_new_hard,
                                     loss_2_cons]
                weight_key_list = ['clean', 'hard', 'noisy', 'w_pseudo', 's_pseudo']
                weight_value_list = [clean_weight_loss, hard_weight_loss, noisy_weight_loss, w_weight_pseudo,
                                     s_weight_pseudo]

                if self.record_history:
                    record_history(self.history_loss_1, loss_1_key_list, loss_1_value_list, epoch, indexes)
                    record_history(self.history_loss_2, loss_2_key_list, loss_2_value_list, epoch, indexes)
                    record_history(self.history_weight, weight_key_list, weight_value_list, epoch, indexes)

            loss_1.backward()
            loss_2.backward()
            self.optimizer.step()
            with torch.no_grad():
                pbar.set_description(
                    'Epoch [%d/%d], Loss1: %.4f, Loss2: %.4f'
                    % (epoch + 1, self.epochs, loss_1.data.item(), loss_2.data.item()))
                wandb.log({"loss_1": loss_1, "loss_2": loss_2, "epoch": epoch})
                wandb.log({key + "_loss_1": value for key, value in zip(loss_1_key_list, loss_1_value_list)})
                wandb.log({key + "_loss_2": value for key, value in zip(loss_2_key_list, loss_2_value_list)})

        if self.adjust_lr == 1:
            if self.optim_type in ['adam', 'adamw']:
                self.adjust_learning_rate(self.optimizer, epoch)
            elif self.optim_type == 'sgd':
                self.scheduler.step()

    def org_mixup_data(self, x, y):
        lam = Beta(torch.tensor(self.alpha), torch.tensor(self.alpha)).sample() if self.alpha > 0 else 1
        index = torch.randperm(x.shape[0], device=self.device)
        if self.mix_type == 'mixup':
            mixed_x = lam * x + (1 - lam) * x[index, :]
            mixed_y = lam * y + (1 - lam) * y[index]
        elif self.mix_type == 'cutmix':
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.shape, lam)
            mixed_x = x.detach().clone()
            mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.shape[-1] * x.shape[-2]))
            mixed_y = lam * y + (1 - lam) * y[index]
        elif self.mix_type == 'fmix':
            mix_method = FMix(decay_power=3, alpha=self.alpha, size=x.shape[2:], max_soft=0.0, reformulate=False)
            mixed_x, lam, index = mix_method(x)
            mixed_y = lam * y + (1 - lam) * y[index]
        else:
            raise NotImplementedError
        return mixed_x, mixed_y, lam, index

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = (W * cut_rat).to(dtype=torch.int32, device=self.device)
        cut_h = (H * cut_rat).to(dtype=torch.int32, device=self.device)

        # uniform
        cx = torch.randint(0, W, (1,), device=self.device)
        cy = torch.randint(0, H, (1,), device=self.device)

        bbx1 = torch.clip(cx - cut_w // 2, 0, W)
        bby1 = torch.clip(cy - cut_h // 2, 0, H)
        bbx2 = torch.clip(cx + cut_w // 2, 0, W)
        bby2 = torch.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], self.beta2)  # Only change beta1

    def save_history(self, file_name):
        torch.save({'weak_lids': self.history_weak_lids.cpu(), 'strong_lids': self.history_strong_lids.cpu(),
                    'pseudo_labels': self.history_labels.cpu(), 'ground_truth_labels': self.ground_truth_label,
                    'loss_1': self.history_loss_1, 'loss_2': self.history_loss_2, 'weight': self.history_weight,
                    'config': self.training_config}, file_name)
