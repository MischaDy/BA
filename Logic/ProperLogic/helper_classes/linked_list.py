class FixedSizeLinkedList:
    class Node:
        def __init__(self, value, next_node=None):
            self.value = value
            self.next_node = next_node

        def get_value(self):
            return self.value

        def get_next(self):
            return self.next_node

        def set_value(self, value):
            self.value = value

        def set_next(self, next_node):
            self.next_node = next_node

    def __init__(self, iterable):
        self.size = len(iterable)
        self.root = self.Node(iterable[0])
        prev_node = self.root
        for item in iterable[1:]:
            new_node = self.Node(item)
            prev_node.set_next(new_node)
            prev_node = new_node

    def insert(self, ind):
        pass
