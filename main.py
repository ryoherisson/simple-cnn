import argparse
from pathlib import Path
import yaml
from  datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from tensorboardX import SummaryWriter
from torchsummary import summary

from utils.dataset import make_datapath_list
from utils.dataset import DataTransforms, Dataset
from models.updater import Updater
from models.metrics.metrics import Metrics
from models.networks.cnn_classifier import CNNClassifier

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configfile', type=str, default='./configs/default.yml')
    args = parser.parse_args()
    return args

def main():
    args = parser()

    ### setup configs ###
    configfile = args.configfile

    with open(configfile) as f:
        configs = yaml.load(f)


    ### setup logs and summary writer ###
    now = datetime.now().isoformat()
    log_dir = Path(configs['log_dir']) / now
    log_dir.mkdir(exist_ok=True, parents=True)

    summary_dir = log_dir / 'tensorboard'
    summary_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(str(summary_dir))


    ### setup GPU or CPU ###
    if configs['n_gpus'] > 0 and torch.cuda.is_available():
        print('CUDA is available! using GPU...')
        device = torch.device('cuda')
    else:
        print('using CPU...')
        device = torch.device('cpu')


    ### Dataset ###
    print('preparing dataset...')

    if configs['dataset'] == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize(configs['img_size'], configs['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(configs['color_mean'], configs['color_std']),
        ])
        train_dataset = datasets.CIFAR10(root=configs['data_root'], train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root=configs['data_root'], train=False, transform=transform, download=True)
    elif configs['dataset'] == 'custom':
        train_transform = DataTransforms(img_size=configs['img_size'], color_mean=configs['color_mean'], color_std=configs['color_std'], phase='train')
        test_transform = DataTransforms(img_size=configs['img_size'], color_mean=configs['color_mean'], color_std=configs['color_std'], phase='test')
        train_img_list, train_lbl_list, test_img_list, test_lbl_list = make_datapath_list(root=configs['data_root'])
        train_dataset = Dataset(train_img_list, train_lbl_list, transform=train_transform)
        test_dataset = Dataset(test_img_list, test_lbl_list, transform=test_transform)
    else:
        raise ValueError('That dataset is not supported')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=configs['batch_size'], shuffle=False, num_workers=8)


    ### Network ###
    print('preparing network...')

    network = CNNClassifier(in_channels=configs['n_channels'], n_classes=configs['n_classes'])

    network = network.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=configs['lr'])

    if configs['resume']:
        # Load checkpoint
        print('==> Resuming from checkpoint...')
        if not Path(configs['resume']).exists():
            assert 'No checkpoint found !'

        ckpt = torch.load(configs['resume'])
        network.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        loss = ckpt['loss']
    else:
        print('==> Building model...')
        start_epoch = 0

    print('model summary: ')
    summary(network, input_size=(configs['n_channels'], configs['img_size'], configs['img_size']))

    if configs["n_gpus"] > 1:
        network = nn.DataParallel(network)


    ### Metrics ###
    metrics = Metrics(n_classes=configs['n_classes'], classes=configs['classes'], writer=writer, log_dir=log_dir)


    ### Train or Test ###
    kwargs = {
        'device': device,
        'network': network,
        'optimizer': optimizer,
        'criterion': criterion,
        'data_loaders': (train_loader, test_loader),
        'metrics': metrics,
        'n_classses': configs['n_classes'],
        'save_ckpt_interval': configs['save_ckpt_interval'],
        'log_dir': log_dir,
    }

    updater = Updater(**kwargs)

    if configs['test']:
        print('mode: test\n')
        updater.test()
    else:
        print('mode: train\n')
        updater.train(n_epochs=configs['n_epochs'], start_epoch=start_epoch)

if __name__ == "__main__":
    main()