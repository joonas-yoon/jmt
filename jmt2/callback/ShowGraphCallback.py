from .Base import Callback

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from IPython.display import display
from ipywidgets import Output


class ShowGraphCallback(Callback):
    def __init__(
        self,
        monitors=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if monitors == None:
            self.monitors = ['train_loss', 'valid_loss']
        else:
            self.monitors = list(monitors)
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(1, 1, 1)
        self.out = Output()
        display(self.out)

    def show_graph(self, log: dict):
        # if self.verbose:
        #     self.message('show_graph')
        try:
            with self.out:
                self.ax.cla()
                self.ax.margins(y=1.0)
                self.ax.set_ylim(bottom=0)
                self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                if log == None:
                    self.ax.plot([], [])
                else:
                    x = range(1, 1+len(log['train_loss']))
                    marker = 'o' if len(x) < 2 else '-'
                    self.ax.plot(x, log['train_loss'], marker)
                    self.ax.plot(x, log['valid_loss'], marker)
                self.out.clear_output(wait=True)
                display(self.figure)
        except KeyboardInterrupt:
            pass

    def before_fit(self, info: dict):
        # if self.verbose:
        #     self.message('before_fit')
        self.show_graph(None)

    def on_train_end(self, epoch: int, info: dict):
        # if self.verbose:
        #     self.message('on_train_end')
        self.show_graph(info)
