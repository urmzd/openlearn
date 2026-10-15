#!/bin/python3.9
from typing import Optional, Union
import sys


class Node:
    def __init__(
        self,
        data,
        previous_node=None,
        left: "Node" = None,
        right: "Node" = None,
        comparator=None,
        data_key=None,
        range_key=None,
        index_key=None,
    ):
        self.data = data
        self.previous_node = previous_node

        self.left = left
        self.right = right
        self.comparator = comparator

        self.data_key = data_key
        self.range_key = range_key
        self.index_key = index_key

        self.height = 0

    def compare(self, y, keying=None):
        _x, _y = (self.data_key, self.data_key) if keying is None else keying

        return self.comparator(self.data, y, keying=(_x, _y))

    def balance_factor(self):
        left_height = 1 if self.left and self.left.height else -1
        right_height = 1 if self.right and self.right.height else -1

        return right_height - left_height

    def rotate_left(self):
        new_parent = self.right
        self.right = new_parent.left
        new_parent.left = self
        self.update()
        new_parent.update()

        return new_parent

    def rotate_right(self):
        new_parent = self.left
        self.left = new_parent.right
        new_parent.right = self
        self.update()
        new_parent.update()

        return new_parent

    def rotate_left_right(self):
        self.left.rotate_left()
        return self.rotate_right()

    def rotate_right_left(self):
        self.right.rotate_right()
        return self.rotate_left()

    def balance(self):
        if self.balance_factor() == -2:
            if self.left.balance_factor() <= 0:
                return self.rotate_right()
            else:
                return self.rotate_left_right()

        if self.balance_factor() == 2:
            if self.right.balance_factor() >= 0:
                return self.rotate_left()
            else:
                return self.rotate_left_right()

        return self

    def get_min_data_key(self):

        curr = self.left

        while curr.left:
            curr = self.left

        return curr.data

    def update_height(self):
        left_height = self.left.height if self.left else -1
        right_height = self.right.height if self.right else -1

        self.height = 1 + max(left_height, right_height)

    def update(self):
        self.update_height()


class AVLTree:
    def __init__(self, comparator=None, range_key=None, data_key=None, index_key=None):
        self.root = None
        self.comparator = comparator
        self.range_key = range_key
        self.data_key = data_key
        self.index_key = index_key

    def insert(self, data, previous_data=None) -> None:
        """
        The following implementation has a bug in it which results
        in duplicates to be deleted.

        For the solutions sake, assume this works correctly.
        For testing, DO NOT ADD DUPLICATES.
        """
        self.root = self._insert(data, previous_data, self.root)

    def _insert(self, data, previous_data=None, parent: Optional[Node] = None):
        if not parent:
            return Node(
                data,
                previous_node=previous_data,
                comparator=self.comparator,
                data_key=self.data_key,
                range_key=self.range_key,
            )

        if parent.compare(data) > 0:
            parent.right = self._insert(data, previous_data, parent.right)

        if parent.compare(data) < 0:
            parent.left = self._insert(data, previous_data, parent.left)

        if not parent.compare(data):
            if parent.compare(data, keying=(self.index_key, self.index_key)) > 0:
                parent.right = self._insert(data, previous_data, parent.right)

            if parent.compare(data, keying=(self.index_key, self.index_key)) < 0:
                parent.left = self._insert(data, previous_data, parent.left)

        parent.update()
        return parent.balance()

    def _in_order(self, parent=Optional[Node]):
        if parent:
            yield from self._in_order(parent.left)

            yield (parent.data)

            yield from self._in_order(parent.right)

    def in_order(self):
        return list(self._in_order(self.root))

    def range_max(self, start: Union[float, int], end: Union[float, int]):
        return self._range_max(start, end, self.root)

    def _range_max(self, start, end, parent: Optional[Node]) -> Optional[Node]:
        """
        The following can be done in O(log(n)) time by
        augmenting this tree to include the highest range at
        within each node.

        The current solution is in O(n) due to poor implementation.
        """

        if not parent:
            return None

        after_start = parent.compare(start, keying=(self.data_key, None)) <= 0
        before_end = parent.compare(end, keying=(self.data_key, None)) >= 0
        in_range = after_start and before_end

        if in_range:
            arr = [
                parent,
                self._range_max(start, end, parent.left),
                self._range_max(start, end, parent.right),
            ]
            return max(filter(lambda x: x, arr), key=lambda x: x.data[self.range_key])

        return None

    def _get_lkios(self, parent):
        if not parent:
            return []

        return [parent.data[self.data_key]] + self._get_lkios(parent.previous_node)

    def get_lkios(self):
        max_root = self.range_max(-float("inf"), float("inf"))
        return list(reversed(self._get_lkios(max_root)))


def compare_fn(x, y, keying):
    _y = y[keying[1]] if keying[1] is not None else y
    _x = x[keying[0]] if keying[0] is not None else x

    return _y - _x


def calculate_cost(S):
    """
    IMPLEMENTATION Bugs:
        1: Duplicates are being deleted.
        2. range_max is being done in O(n) time instead of O(log(n))

    For more: Take a look at AVLTree.insert() function and
        AVLTree.range_max() function.
    """
    tree = AVLTree(data_key=0, range_key=1, index_key=2, comparator=compare_fn)
    length = len(S)

    for i in range(length):
        range_max: Optional[Node] = tree.range_max(S[i] - 3, S[i])

        if range_max:
            y_h = range_max.data[1]
            tree.insert((S[i], y_h + 1, i), previous_data=range_max)
        else:
            tree.insert((S[i], 1, i))

    return tree.get_lkios()


if __name__ == "__main__":
    S = [1, 9, 2, 3, 7, 6, 5]

    if len(sys.argv) >= 2:
        S = list(map(lambda x: int(x), sys.argv[1:]))

    print("The LIKOS is:")
    print(calculate_cost(S))
