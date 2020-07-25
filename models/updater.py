from tqdm import tqdm

from collections import OrderedDict

import torch


class Updater(object):
    def __init__(self, **kwargs):
        self.device = kwargs['device']
        self.network = kwargs['network']
        self.optimizer = kwargs['optimizer']
        self.criterion = kwargs['criterion']
        self.train_loader, self.test_loader = kwargs['data_loaders']
        self.n_classes = kwargs['n_classes']

    def train(self, n_epochs):

        for epoch in range(n_epochs):
            self.network.train()

            train_loss = 0
            n_correct = 0
            n_total = 0

            with tqdm(self.train_loader, ncols=100) as pbar:
                for idx, (inputs, targets) in enumerate(pbar):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.network(inputs)

                    loss = self.criterion(outputs, targets)

                    loss.backward()

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    train_loss += loss.item()

                    pred = outputs.argmax(axis=1)
                    n_total += targets.size(0)
                    n_correct += (pred == targets).sum().item()

                    pbar.set_postfix(OrderedDict(
                        epoch="{:>10}".format(epoch),
                        loss="{:.4f}".format(train_loss),
                        acc="{:.4f}".format(100.0 * n_correct / n_total)))

            print(f'train loss: {train_loss}')
            print(f'train accuracy: {100.0 * n_correct / n_total}')

            self.test(epoch)

    def test(self, epoch):
        self.network.eval()
    
        test_loss = 0
        n_correct = 0
        n_total = 0

        with torch.no_grad():
            with tqdm(self.test_loader, ncols=100) as pbar:
                    for idx, (inputs, targets) in enumerate(pbar):

                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)

                        outputs = self.network(inputs)

                        loss = self.criterion(outputs, targets)

                        self.optimizer.zero_grad()

                        test_loss += loss.item()

                        pred = outputs.argmax(axis=1)
                        n_total += targets.size(0)
                        n_correct += (pred == targets).sum().item()

                        pbar.set_postfix(OrderedDict(
                            epoch="{:>10}".format(epoch),
                            loss="{:.4f}".format(test_loss),
                            acc="{:.4f}".format(100.0 * n_correct / n_total)))

            print(f'test loss: {test_loss}')
            print(f'test accuracy: {100.0 * n_correct / n_total}\n')


    def _save_ckpt(self):
        pass