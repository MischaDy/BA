import operator
from collections import UserList

from Logic.ProperLogic.misc_helpers import enumerate_descending, identity


# TODO: Replace by appropriate LinkedList!
class SortedList(UserList):
    """
    Sorted descending
    """
    def __init__(self, items=None, max_size=None, key=identity, order=None):
        if items is None:
            items = []
        if order is None:
            self.order = operator.lt
        super().__init__(items)
        self.max_size = max_size if max_size is not None else float('inf')
        self.key = key

    def append(self, item):
        self.add(item)

    def add(self, new_item):
        new_item_val = self.key(new_item)
        try:
            if self.order(self.data[-1], new_item_val):
                return
        except IndexError:
            self.data.append(new_item)
            return

        ind = 0
        for ind, item in enumerate_descending(self.data[:-1]):
            if self.order(self.key(item), new_item_val):
                break
        # previous ind in order, not in size
        prev_ind = ind + 1
        self.data.insert(prev_ind, new_item)
        if len(self.data) > self.max_size:
            self.pop()
