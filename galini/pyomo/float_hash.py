# Copyright 2017 Francesco Ceccon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Classes to hash floating point numbers."""
import abc
from suspect.math import (make_number, almosteq, almostlte) # pylint: disable=no-name-in-module


class FloatHasher(abc.ABC):
    """Hashes floating point numbers."""
    @abc.abstractmethod
    def hash(self, number):
        """Return `f` hash."""
        raise NotImplementedError('hash')


class BTreeFloatHasher(FloatHasher):
    """A floating point hasher that keeps all seen floating
    point numbers ina binary tree.

    Good if the unique values of the floating point numbers in
    the problem are relatively few.
    """

    class Node(object):
        """BTree Node"""
        def __init__(self, num: float, hash_: int) -> None:
            self.num = num
            self.hash = hash_
            self.left = None
            self.right = None

    def __init__(self) -> None:
        self.root = None
        self.node_count = 0

    def hash(self, number):
        number = make_number(number)
        if self.root is None:
            self.root = self._make_node(number)
            return self.root.hash

        curr_node = self.root
        while True:
            if almosteq(number, curr_node.num):
                return curr_node.hash
            elif almostlte(number, curr_node.num):
                if curr_node.left is None:
                    new_node = self._make_node(number)
                    curr_node.left = new_node
                    return new_node.hash
                else:
                    curr_node = curr_node.left
            else:
                if curr_node.right is None:
                    new_node = self._make_node(number)
                    curr_node.right = new_node
                    return new_node.hash
                else:
                    curr_node = curr_node.right

    def _make_node(self, number):
        node = self.Node(number, self.node_count)
        self.node_count += 1
        return node


class RoundFloatHasher(FloatHasher):
    """A float hasher that hashes floats up to the n-th
    decimal place.
    """
    def __init__(self, n=2):
        self.n = 10**n

    def hash(self, number):
        return hash(int(number * self.n))
