import argparse
from utils import load_config, get_log_name, set_seed, save_results, \
    print_config, get_avg_performance, update_best_performance, add_performance
from datasets import cifar_dataloader
import algorithms
import numpy as np
import nni
import time
import pickle
import json
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    '-c',
                    type=str,
                    default='./configs/colearning.py',
                    help='The path of config file.')
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--dataset', type=str, default='cifar-10')
parser.add_argument('--root', type=str, default='/data/CIFAR10')
parser.add_argument('--save_path', type=str, default='./log/')
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--noise_type', type=str, default='sym')
parser.add_argument('--percent', type=float, default=0.2)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--performance_path', type=str, default='test_performance')
parser.add_argument('--noise_path', type=str, default='../data/CIFAR-10_human.pt')
args = parser.parse_args()


def main():
    tuner_params = nni.get_next_parameter()
    config = load_config(args.config, _print=False)
    config.update(tuner_params)
    config['dataset'] = args.dataset
    config['root'] = args.root
    config['gpu'] = args.gpu
    config['noise_type'] = args.noise_type
    config['percent'] = args.percent
    config['seed'] = args.seed
    config['num_classes'] = args.num_classes
    config['momentum'] = args.momentum
    config['performance_path'] = args.performance_path
    config['noise_path'] = args.noise_path
    if 'noise_path' not in config:
        config['noise_path'] = None
    print_config(config)
    wandb.init(
        # set the wandb project where this run will be logged
        project="CoLafier",
        name=f"{config['algorithm']}-{config['dataset']}-{config['noise_type']}-{config['percent']}-{config['seed']}",
        # track hyperparameters and run metadata
        config=config
    )
    wandb.run.name = wandb.run.name + '-' + wandb.run.id
    set_seed(config['seed'])

    if config['algorithm'] == 'DISC':
        model = algorithms.DISC(config,
                                input_channel=config['input_channel'],
                                num_classes=config['num_classes'])
        train_mode = 'train_index'

    elif config['algorithm'] == 'colearning':
        model = algorithms.Colearning(config,
                                      input_channel=config['input_channel'],
                                      num_classes=config['num_classes'])
        train_mode = 'train'

    elif config['algorithm'] == 'JointOptimization':
        model = algorithms.JointOptimization(
            config,
            input_channel=config['input_channel'],
            num_classes=config['num_classes'])
        train_mode = 'train_index'

    elif config['algorithm'] == 'GJS':
        model = algorithms.GJS(config,
                               input_channel=config['input_channel'],
                               num_classes=config['num_classes'])
        train_mode = 'train_index'

    elif config['algorithm'] == 'ELR':
        model = algorithms.ELR(config,
                               input_channel=config['input_channel'],
                               num_classes=config['num_classes'])
        train_mode = 'train_index'

    elif config['algorithm'] == 'PENCIL':
        model = algorithms.PENCIL(config,
                                  input_channel=config['input_channel'],
                                  num_classes=config['num_classes'])
        train_mode = 'train_index'

    elif config['algorithm'] == 'CoLafier':
        model = algorithms.CoLafier(config,
                                    input_channel=config['input_channel'],
                                    num_classes=config['num_classes'])
        train_mode = 'train_index'
    else:
        model = algorithms.__dict__[config['algorithm']](
            config,
            input_channel=config['input_channel'],
            num_classes=config['num_classes'])
        train_mode = 'train_single'
        if config['algorithm'] == 'StandardCETest':
            train_mode = 'train_index'
        elif config['algorithm'] == 'StandardCE':
            train_mode = 'train_index_regular'

    if 'cifar' in config['dataset']:
        dataloaders = cifar_dataloader(cifar_type=config['dataset'],
                                       root=config['root'],
                                       batch_size=config['batch_size'],
                                       num_workers=config['num_workers'],
                                       noise_type=config['noise_type'],
                                       percent=config['percent'],
                                       noise_path=config['noise_path'])
        trainloader, testloader = dataloaders.run(
            mode=train_mode), dataloaders.run(mode='test')

    num_test_images = len(testloader.dataset)

    start_epoch = 0
    epoch = 0

    # evaluate models with random weights
    test_performance = get_avg_performance(model.evaluate(testloader))
    print('Epoch [%d/%d] Test Accuracy on the %s test images: %.4f' %
          (epoch, config['epochs'], num_test_images, test_performance['acc']))

    acc_list, acc_all_list = [], []
    best_dict, performance_list_dict = {}, {}

    # loading training labels
    if config['algorithm'] in ['DISC', 'StandardCETest', 'CoLafier']:
        if 'cifar' in config['dataset']:
            model.get_labels(trainloader)
        model.weak_labels = model.labels.detach().clone()
        print('The labels are loaded!!!')
    else:
        print("Skip labels loading procedure!!!")

    since = time.time()
    best_dict['since'] = since
    for epoch in range(start_epoch, config['epochs']):
        # train
        model.train(trainloader, epoch)
        # evaluate
        test_performance = get_avg_performance(model.evaluate(testloader))

        best_dict = update_best_performance(best_dict, epoch, test_performance)

        print(
            'Epoch [%d/%d] Test Accuracy on the %s test images: %.4f %%' %
            (epoch + 1, config['epochs'], num_test_images, test_performance['acc']))

        if epoch >= config['epochs'] - 10:
            acc_list.extend([test_performance['acc']])

        performance_list_dict = add_performance(performance_list_dict, test_performance)

    time_elapsed = time.time() - since
    total_min = time_elapsed // 60
    hour = total_min // 60
    min = total_min % 60
    sec = time_elapsed % 60
    acc_all_list = performance_list_dict['acc']
    config['training_time'] = time_elapsed
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        hour, min, sec))

    if config['save_result']:
        config['algorithm'] = config['algorithm'] + args.model_name
        acc_np = np.array(acc_list)
        nni.report_final_result(acc_np.mean())
        jsonfile = get_log_name(config, path=args.save_path)
        with open(jsonfile.replace('.json', '.pkl'), 'wb') as fp:
            pickle.dump(performance_list_dict, fp)
        np.save(jsonfile.replace('.json', '.npy'), np.array(acc_all_list))
        if 'record_history' in config.keys():
            if config['record_history']:
                model.save_history(jsonfile.replace('.json', '.pt'))
        save_results(config=config,
                     last_ten=acc_np,
                     best_dict=best_dict,
                     jsonfile=jsonfile)
        with open(config['performance_path'], 'a+') as outfile:
            all_performance_dict = config
            all_performance_dict.update(best_dict)
            all_performance_dict.update(performance_list_dict)
            outfile.write(json.dumps(all_performance_dict) + '\n')


if __name__ == '__main__':
    main()
