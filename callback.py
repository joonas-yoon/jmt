import os
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from ipywidgets import Output


class Callback:
    def __init__(self, verbose=False):
        self.verbose = verbose
    def before_fit(self):
        pass
    def message(self, msg, prefix=''):
        if self.verbose:
            print(f'[{self.__class__.__name__}{prefix}] {msg}')
    def on_train_end(self):
        pass


class CheckpointCallback(Callback):
    def __init__(
        self,
        name,
        path='./',
        metric=np.less,
        min_delta=0.0,
        load_from_checkpoint=False, # str: filepath
        **kwargs
    ):
        super().__init__(**kwargs)
        self.name = name
        self.path = path
        self.filepath = os.path.join(self.path, self.name)
        self.min_delta = min_delta
        self.metric = metric
        self.load_from_checkpoint = load_from_checkpoint
        self.best = np.Inf

    def load_model(self, model: nn.Module, filepath: str):
        file_exists = os.path.exists(filepath)
        if self.verbose:
            if file_exists: self.message(f'model loaded from {self.filepath}')
            else: self.message(f'Failed to load model from {self.filepath}')
        if file_exists:
            model.load_state_dict(torch.load(self.filepath))

    def before_fit(self, model: nn.Module):
        lfc = self.load_from_checkpoint
        if isinstance(lfc, bool):
            if lfc == True:
                self.load_model(model, self.filepath)
        elif isinstance(lfc, str):
            self.load_model(model, lfc)
        else:
            self.message('Disallowed checkpoint parameter:', type(lfc), lfc)

    # update and save model if loss is less than before
    def on_train_end(self, model: nn.Module, loss: float):
        if self.metric(self.best, loss + self.min_delta): return None
        self.message(f'update best: {loss}', prefix=f'({self.filepath})')
        self.best = loss
        torch.save(model.state_dict(), self.filepath)


class EarlyStopCallback(Callback):
    def __init__(
        self,
        patience=5,
        metric=np.less,
        min_delta=0.0,
        max_seconds=math.inf,
        last_k_epochs=3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.max_seconds = max_seconds
        self.last_k_epochs = last_k_epochs
        self.bad_epochs = 0

    # diff two gradient of mean
    def diff(self, losses):
        k = self.last_k_epochs
        if k + 1 > len(losses): return False
        before, after = losses[-k-1:-1], losses[-k:]
        before, after = np.diff(before).mean(), np.diff(after).mean()
        # return self.metric(before, after + self.min_delta)
        return self.metric(after, self.min_delta)

    def on_train_end(self, epoch: int, time_elapsed: int, info: dict):
        if time_elapsed >= self.max_seconds: return True
        if self.diff(info['train_loss']) or self.diff(info['valid_loss']):
            self.bad_epochs += 1
        else:
            self.bad_epochs = 0
        self.message(f'remain {self.patience - self.bad_epochs} patience(s)')
        return self.bad_epochs >= self.patience


class ShowGraphCallback(Callback):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fig = plt.figure()

        self.out = Output()
        display.display(self.out)

    def show_graph(self, log: dict):
        with self.out:
            plt.clf()
            plt.ylim(bottom=0)
            if log == None:
                plt.plot([0], [0])
                plt.show()
            else:
                x = range(1, 1+len(log['train_loss']))
                plt.plot(x, log['train_loss'])
                plt.plot(x, log['valid_loss'])
                plt.show()
            display.clear_output(wait=True)

    def before_fit(self, info: dict):
        self.show_graph(None)

    def on_train_end(self, epoch: int, info: dict):
        self.show_graph(info)
