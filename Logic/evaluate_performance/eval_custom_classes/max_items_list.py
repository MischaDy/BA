from collections import UserList


class SortedList(UserList):
    def __init__(self, items=None, max_size=None, key=max):
        if items is None:
            items = []
        super().__init__(items)
        self.max_size = max_size if max_size is not None else float('inf')
        self.key = key

    def append(self, item):

        if len(self.data) > self.max_size:
            self.pop()
