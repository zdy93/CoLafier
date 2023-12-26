# CoLafier
This repository is the official implementation of the **SDM2024** paper "CoLafier: Collaborative Noisy Label Purifier With Local Intrinsic Dimensionality Guidance". Some code scripts are adapted from [DISC](https://github.com/JackYFL/DISC/tree/main).

## Requirements
### Language
* Python3 == 3.9.12
### Modules
* wandb==0.10.11
* torch==2.0.0+cu118
* addict==2.4.0
* torchvision==0.15.1+cu118
* tqdm==4.64.1
* scikit-learn==1.2.0
* matplotlib==3.6.2
* numpy==1.24.2
* Pillow==9.4.0
These packages can be installed directly by running the following command:
```
pip install -r requirements.txt
```
## Datasets
### CIFAR-10
CIFAR-10 dataset can be downloaded from [link](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz). 
### CIFAR-10N
CIFAR-10N labels are provided at [cifar-10-100n](https://github.com/UCSC-REAL/cifar-10-100n/tree/main). In our paper, we used the "Aggregate", "Random1", and "Worse" labels.

If you want to use one of the datasets, please download it into your data directory and change the data path in bash scripts (see the following section).

## Training
### CIFAR-10 with symmetric or asymmetric noise
```bash
noise_type=${1:-'sym'} # 'sym' or 'asym' 
gpuid=${2:-'0'}
seed=${3:-'1'}
save_path=${4:-'./logs/'}
data_path=${5:-'../data'} # path to the CIFAR-10 data
config_path=${6:-'./configs/colafier_sym_asym.py'}
dataset=${7:-'cifar-10'}
num_classes=${8:-10}
noise_rate=${9:-0.4}
performance_path=${10:-'test_performance'}

python main.py
  -c=$config_path
  --save_path=$save_path
  --noise_type=$noise_type
  --seed=$seed --gpu=$gpuid
  --percent=$noise_rate
  --dataset=$dataset
  --num_classes=$num_classes
  --root=$data_path
  --performance_path=$performance_path
  --noise_path=$noise_path
```

### CIFAR-10 with instance-dependent noise or CIFAR-10N
```bash
noise_type=${1:-'ins'} # 'ins' or 'aggre_label' or 'worse_label' or 'random_label1' 
gpuid=${2:-'0'}
seed=${3:-'1'}
save_path=${4:-'./logs/'}
data_path=${5:-'../data'} # path to the CIFAR-10/CIFAR-10N data
config_path=${6:-'./configs/colafier_ins_real.py'}
dataset=${7:-'cifar-10'} 
num_classes=${8:-10}
noise_rate=${9:-0.4} # set 0.0 for CIFAR-10N
performance_path=${10:-'test_performance'}

python main.py
  -c=$config_path
  --save_path=$save_path
  --noise_type=$noise_type
  --seed=$seed --gpu=$gpuid
  --percent=$noise_rate
  --dataset=$dataset
  --num_classes=$num_classes
  --root=$data_path
  --performance_path=$performance_path
  --noise_path=$noise_path
```

