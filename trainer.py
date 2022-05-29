from datetime import datetime
import torch
import numpy as np
import gc as garbage
from tqdm.notebook import tqdm as tqdm_nb
from callback import EarlyStopCallback, CheckpointCallback, ShowGraphCallback

class Trainer:
    def __init__(
        self,
        loss_function,
        optimizer,
        max_epochs,
        accelerator='cpu',
        min_epochs=1,
        progress=False,
        batch_parser=None,
        logging=True,
        verbose=False,
        callbacks=[],
        **kwargs
    ):
        self.log = {}
        self.criterion = loss_function
        self.optimizer = optimizer
        self.device = accelerator
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.has_progress = progress
        self.has_logging = logging
        self.verbose = verbose
        self.pbar = None
        self.batch_parser = batch_parser if batch_parser else Trainer.default_parser
        self.start_time = datetime.now()
        self.callbacks = callbacks

    def default_parser(batch):
        x, y = batch
        return x, y

    def set_parser(self, function):
        self.batch_parser = function

    def logging(self, name, value):
        if not self.has_logging: pass
        if not name in self.log:
            self.log[name] = []
        self.log[name].append(value)

    def step_loop(self, model, dataloader, is_train=True):
        torch.cuda.synchronize()
        model.to(self.device)
        if is_train:
            model.train()
        else:
            model.eval()
        loss_sum = 0
        acc_sum = 0
        counter = 0
        for batch in dataloader:
            counter += 1
            x, y = self.batch_parser(batch)
            x, y = x.to(self.device), y.to(self.device)
            y_hat = model(x).to(self.device)
            loss = self.criterion(y_hat, y)
            _, y_pred = torch.max(y_hat.data, 1)
            acc = (y_pred == y).sum()
            loss_sum += loss.item()
            acc_sum += acc.item()
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if type(None) != type(self.pbar):
                self.pbar.update(1)
        loss_sum /= counter
        acc_sum /= counter
        return loss_sum, acc_sum

    def train(self, model, dataloader):
        return self.step_loop(model, dataloader, True)

    def valid(self, model, dataloader):
        return self.step_loop(model, dataloader, True)

    def test(self, model, dataloader):
        return self.step_loop(model, dataloader, False)

    def fit(self, model, load_dataloaders):
        if self.has_progress:
            self.progress_start().set_description('on ready to fit')
        self.start_time = datetime.now()

        # torch.cuda.empty_cache()
        garbage.collect()

        for cb in self.callbacks:
            if issubclass(type(cb), CheckpointCallback):
                cb.before_fit(model)
            elif issubclass(type(cb), ShowGraphCallback):
                cb.before_fit(self.log)

        for e in range(self.max_epochs):
            train_loader, valid_loader = load_dataloaders()

            if self.has_progress:
                self.pbar.reset(len(train_loader) + len(valid_loader))
                self.pbar.set_description(f'[epoch={e+1}/{self.max_epochs}] train')

            train_loss, train_acc = self.train(model, train_loader)
            self.logging('train_loss', train_loss)
            self.logging('train_acc', train_acc)

            if self.has_progress:
                self.pbar.set_description(f'[epoch={e+1}/{self.max_epochs}] valid')

            valid_loss, valid_acc = self.valid(model, valid_loader)
            self.logging('valid_loss', valid_loss)
            self.logging('valid_acc', valid_acc)

            time_elapsed = (datetime.now() - self.start_time)
            msg = 'train/valid loss: {:.05f}/{:.05f}'.format(train_loss, valid_loss)

            if self.has_progress: 
                self.pbar.set_postfix_str(msg)

            if self.verbose:
                print(f'[Trainer(epoch={e+1})]', str(time_elapsed), msg)

            early_stop = False
            for cb in self.callbacks:
                if issubclass(type(cb), ShowGraphCallback):
                    cb.on_train_end(e, self.log)
                elif issubclass(type(cb), CheckpointCallback):
                    cb.on_train_end(model, np.mean([train_loss, valid_loss]))
                elif issubclass(type(cb), EarlyStopCallback):
                    early_stop |= cb.on_train_end(e, time_elapsed.seconds, self.log)
                    if early_stop: break

            if e < self.min_epochs:
                continue

            if early_stop:
                break

            garbage.collect()

        self.progress_end()

    def progress_start(self, total=None, desc=None):
        if type(None) == type(self.pbar):
            self.pbar = tqdm_nb()
        self.pbar.reset(total)
        self.pbar.set_description(desc)
        return self.pbar

    def progress_end(self):
        self.pbar.close()
        self.pbar = None
