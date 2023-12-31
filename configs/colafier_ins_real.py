algorithm = 'CoLafier'
# dataset param
dataset = 'cifar-10'
input_channel = 3
num_classes = 10
root = '/data/CIFAR10'
noise_type = 'asym'
percent = 0.4
seed = 1
# model param
model1_type = 'resnet34'
model2_type = 'resnet34'
# train param
batch_size = 128
lr = 0.001
beta2 = 0.999
epochs = 200
num_workers = 4
gpu = '0'
exponent = 1
adjust_lr = 0
epoch_decay_start = 50
optimizer = 'adamw'
no_decay = False
weight_decay = 1e-3
milestones=[80, 160]
# LID
lid_k = 20
lid_metric = 'l2'
warmup_epoch = 15
keep_threshold_init = 0.1
keep_threshold_end = 0.1
num_keep_gradual = 30
keep_after_quant = False
low_quant_init = 0.001
low_quant_end = 0.001
high_quant_init = 0.05
high_quant_end = 1.0
num_quant_gradual = 30
loss_low_quant_init = 0.001
loss_low_quant_end = 0.001
loss_high_quant_init = 0.50
loss_high_quant_end = 1.0
num_loss_quant_gradual = 30
consistency_loss = True
consider_diff = True
mixup_loss = True
alpha = 5.0
early_stop = False
merge_concat = False
activation = None
new_label_loss = True
lambda_clean = 1
lambda_hard = 1
lambda_mix = 1
lambda_new = 0.5
lambda_cons = 10
mix_type = 'cutmix'
embedding_type = 'regular'
u_lid = False
# result param
save_result = True
record_history = False
