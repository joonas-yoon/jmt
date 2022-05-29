import os
import torch
import torch.nn as nn
import numpy as np

from .Base import Callback


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
