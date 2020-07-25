from tqdm import tqdm

from collections import OrderedDict

import torch
import torch.nn as nn


class CNNClassifier(object):
    def __init__(self, **kwargs):
        self.device = kwargs['device']
        self.network = kwargs['network']
        self.optimizer = kwargs['optimizer']
        self.criterion = kwargs['criterion']
        self.train_loader, self.test_loader = kwargs['data_loaders']
        self.metrics = kwargs['metrics']
        self.save_ckpt_interval = kwargs['save_ckpt_interval']
        self.ckpt_dir = kwargs['ckpt_dir']

    def train(self, n_epochs, start_epoch=0):

        best_accuracy = 0

        for epoch in range(start_epoch, n_epochs):
            print(f'---------------------- Epoch: {epoch} ----------------------')
            print('# train:')
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

                    accuracy = 100.0 * n_correct / n_total

                    self.metrics.update(
                        preds=pred.cpu().detach().clone(),
                        targets=targets.cpu().detach().clone(),
                        loss=train_loss / n_total,
                        accuracy=accuracy,
                    )

                    ### logging train loss and accuracy
                    pbar.set_postfix(OrderedDict(
                        epoch="{:>10}".format(epoch),
                        loss="{:.4f}".format(train_loss / n_total),
                        acc="{:.4f}".format(accuracy)))

            print(f'train loss: {train_loss / n_total}')
            print(f'train accuracy: {accuracy}')

            self.metrics.calc_metrics(epoch, mode='train')
            self.metrics.init_cmx()

            if epoch % self.save_ckpt_interval == 0:
                self._save_ckpt(epoch, train_loss/(idx+1))

            ### test
            print('# test:')
            test_accuracy = self.test(epoch)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                self._save_ckpt(epoch, train_loss/(idx+1), mode='best')

    def test(self, epoch):
        self.network.eval()
    
        test_loss = 0
        n_correct = 0
        n_total = 0
        preds_t = torch.tensor([])

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

                        accuracy = 100.0 * n_correct / n_total

                        self.metrics.update(
                            preds=pred.cpu().detach().clone(),
                            targets=targets.cpu().detach().clone(),
                            loss=test_loss / n_total,
                            accuracy=accuracy,
                        )

                        ### logging test loss and accuracy
                        pbar.set_postfix(OrderedDict(
                            epoch="{:>10}".format(epoch),
                            loss="{:.4f}".format(test_loss / n_total),
                            acc="{:.4f}".format(accuracy)))

            print(f'test loss: {test_loss / n_total}')
            print(f'test accuracy: {accuracy}\n')

            self.metrics.calc_metrics(epoch, mode='test')
            self.metrics.init_cmx()

        return accuracy

    def _save_ckpt(self, epoch, loss, mode=None, zfill=4):
        if isinstance(self.network, nn.DataParallel):
            network = self.network.module
        else:
            network = self.network

        if mode == 'best':
            ckpt_path = self.ckpt_dir / 'best_acc_ckpt.pth'
        else:
            ckpt_path = self.ckpt_dir / f'epoch{str(epoch).zfill(zfill)}_ckpt.pth'

        torch.save({
            'epoch': epoch,
            'network': network,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, ckpt_path)