import math
import numpy as np
from .Base import Callback

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
        return self.metric(before, after + self.min_delta)

    def on_train_end(self, epoch: int, time_elapsed: int, info: dict):
        if time_elapsed >= self.max_seconds: return True
        if self.diff(info['train_loss']) or self.diff(info['valid_loss']):
            self.bad_epochs += 1
            if self.verbose:
                self.message(f'remain {self.patience - self.bad_epochs} patience(s)')
        else:
            self.bad_epochs = 0
        return self.bad_epochs >= self.patience
