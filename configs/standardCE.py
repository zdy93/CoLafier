algorithm = 'StandardCE'
# dataset param
dataset = 'cifar-10'
input_channel = 3
num_classes = 10
root = '/data/CIFAR10'
noise_type = 'sym'
percent = 0.2
seed = 1
loss_type = 'ce'
# model param
model1_type = 'resnet34'
model2_type = 'none'
# train param
gpu = '1'
batch_size = 128
lr = 0.001
epochs = 300
num_workers = 4
adjust_lr = 1
epoch_decay_start = 80
optimizer = 'adam'
weight_decay = 1e-3
milestones=[80, 160]
# result param
save_result = True