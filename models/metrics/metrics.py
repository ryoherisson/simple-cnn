from pathlib import Path
import csv

import numpy as np
import pandas as pd

import torch

class Metrics(object):
    def __init__(self, n_classes, classes, epsilon=1e-12, log_dir='./logs'):
        self.n_classes = n_classes
        self.classes = classes
        self._init_cmx()
        self.epsilon = epsilon
        self.loss = 0

        self.log_dir = log_dir

        self.metrics_dir = Path(self.log_dir) / 'metrics'
        self.metrics_dir.mkdir(exist_ok=True)


    def update(self, preds, targets, loss, accuracy):
        stacked = torch.stack((targets, preds), dim=1)

        for p in stacked:
            tl, pl = p.tolist()
            self.__cmx[tl, pl] = self.__cmx[tl, pl] + 1

        self.loss = loss
        self.accuracy = accuracy

    def calc_metrics(self, epoch, mode='train'):
        tp = torch.diag(self.__cmx).to(torch.float32)
        fp = (self.__cmx.sum(axis=1) - torch.diag(self.__cmx)).to(torch.float32)
        fn = (self.__cmx.sum(axis=0) - torch.diag(self.__cmx)).to(torch.float32)

        self.precision = tp / (tp + fp + self.epsilon)
        self.recall = tp / (tp + fn + self.epsilon)
        self.f1score = tp / (tp + 0.5 * (fp + fn))

        self.logging()
        self.save_csv(epoch, mode)
        self._init_cmx()

    def _init_cmx(self):
        """Initialize Confusion Matrix tensor with shape (n_classes, n_classes)
        """
        self.__cmx = torch.zeros(self.n_classes, self.n_classes, dtype=torch.int64)

    def logging(self):

        df = pd.DataFrame(index=self.classes)
        df['precision'] = self.precision.tolist()
        df['recall'] = self.recall.tolist()
        df['f1score'] = self.f1score.tolist()

        print(f'metrics values per classes: \n{df}\n')

        print(f'precision: {self.precision.mean()}')
        print(f'recall: {self.recall.mean()}')
        print(f'micro f1score: {self.f1score.mean()}\n')

    def save_csv(self, epoch, mode):

        csv_path = self.metrics_dir / f'{mode}_metrics.csv'

        if not csv_path.exists():
            with open(csv_path, 'w') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(['epoch', f'{mode} loss', f'{mode} accuracy',
                                    f'{mode} precision', f'{mode} recall', f'{mode} micro f1score'])

        with open(csv_path, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, self.loss, self.accuracy, 
                                self.precision.mean().item(), self.recall.mean().item(), self.f1score.mean().item()])