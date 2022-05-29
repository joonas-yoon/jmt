from .Base import Callback

import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from ipywidgets import Output


class ShowGraphCallback(Callback):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(1, 1, 1)
        self.out = Output()
        display(self.out)

    def show_graph(self, log: dict):
        # if self.verbose:
        #     self.message('show_graph')
        with self.out:
            # self.figure.clf()
            # self.ax.set_ylim(top=None, bottom=0)
            self.ax.cla()
            if log == None:
                self.ax.plot([0], [0], 'o')
            else:
                x = range(1, 1+len(log['train_loss']))
                self.ax.plot(x, log['train_loss'], '-')
                self.ax.plot(x, log['valid_loss'], '--')
            display(self.figure)
            clear_output(wait=True)
            plt.pause(0.01)

    def before_fit(self, info: dict):
        # if self.verbose:
        #     self.message('before_fit')
        self.show_graph(None)

    def on_train_end(self, epoch: int, info: dict):
        # if self.verbose:
        #     self.message('on_train_end')
        self.show_graph(info)
