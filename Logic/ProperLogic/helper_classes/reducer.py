from functools import reduce


class Reducer:
    def __init__(self, func, default):
        self.func = func
        self.default = default
        self.state = self.default

    def __call__(self, *args):
        try:
            # args consists of individual items
            self.process(*args)
        except TypeError:
            # args probably consists of iterable
            self.process_iterable(args[0])

    def process(self, *args):
        self.state = reduce(self.func, args, self.state)

    def process_iterable(self, iterable):
        self.state = reduce(self.func, iterable, self.state)

    def get_state(self):
        return self.state

    def reset(self):
        self.state = self.default


class MaxReducer(Reducer):
    def __init__(self, default=float('-inf')):
        super().__init__(max, default)
