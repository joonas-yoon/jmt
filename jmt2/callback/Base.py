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
