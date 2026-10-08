# MIT License
#
# Copyright (c) 2018-2024 Tskit Developers
# Copyright (C) 2017 University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Tree position utilities used by the HMM test suite.
"""

import dataclasses

FORWARD = 1
REVERSE = -1


@dataclasses.dataclass
class Interval:
    left: float
    right: float

    def __iter__(self):
        yield self.left
        yield self.right


@dataclasses.dataclass
class EdgeRange:
    start: int
    stop: int
    order: list


class TreeIndexes:
    def __init__(self, ts):
        self.ts = ts
        self.index = -1
        self.direction = 0
        self.interval = Interval(0, 0)
        self.in_range = EdgeRange(0, 0, None)
        self.out_range = EdgeRange(0, 0, None)

    def __str__(self):
        s = f"index: {self.index}\ninterval: {self.interval}\n"
        s += f"direction: {self.direction}\n"
        s += f"in_range: {self.in_range}\n"
        s += f"out_range: {self.out_range}\n"
        return s

    def assert_equal(self, other):
        assert self.index == other.index
        assert self.direction == other.direction
        assert self.interval == other.interval

    def set_null(self):
        self.index = -1
        self.interval.left = 0
        self.interval.right = 0

    def next(self):  # NOQA: A003
        M = self.ts.num_edges
        breakpoints = self.ts.breakpoints(as_array=True)
        left_coords = self.ts.edges_left
        left_order = self.ts.indexes_edge_insertion_order
        right_coords = self.ts.edges_right
        right_order = self.ts.indexes_edge_removal_order

        if self.index == -1:
            self.interval.right = 0
            self.out_range.stop = 0
            self.in_range.stop = 0
            self.direction = FORWARD

        if self.direction == FORWARD:
            left_current_index = self.in_range.stop
            right_current_index = self.out_range.stop
        else:
            left_current_index = self.out_range.stop + 1
            right_current_index = self.in_range.stop + 1

        left = self.interval.right

        j = right_current_index
        self.out_range.start = j
        while j < M and right_coords[right_order[j]] == left:
            j += 1
        self.out_range.stop = j
        self.out_range.order = right_order

        j = left_current_index
        self.in_range.start = j
        while j < M and left_coords[left_order[j]] == left:
            j += 1
        self.in_range.stop = j
        self.in_range.order = left_order

        self.direction = FORWARD
        self.index += 1
        if self.index == self.ts.num_trees:
            self.set_null()
        else:
            self.interval.left = left
            self.interval.right = breakpoints[self.index + 1]
        return self.index != -1

    def prev(self):
        M = self.ts.num_edges
        breakpoints = self.ts.breakpoints(as_array=True)
        right_coords = self.ts.edges_right
        right_order = self.ts.indexes_edge_removal_order
        left_coords = self.ts.edges_left
        left_order = self.ts.indexes_edge_insertion_order

        if self.index == -1:
            self.index = self.ts.num_trees
            self.interval.left = self.ts.sequence_length
            self.in_range.stop = M - 1
            self.out_range.stop = M - 1
            self.direction = REVERSE

        if self.direction == REVERSE:
            left_current_index = self.out_range.stop
            right_current_index = self.in_range.stop
        else:
            left_current_index = self.in_range.stop - 1
            right_current_index = self.out_range.stop - 1

        right = self.interval.left

        j = left_current_index
        self.out_range.start = j
        while j >= 0 and left_coords[left_order[j]] == right:
            j -= 1
        self.out_range.stop = j
        self.out_range.order = left_order

        j = right_current_index
        self.in_range.start = j
        while j >= 0 and right_coords[right_order[j]] == right:
            j -= 1
        self.in_range.stop = j
        self.in_range.order = right_order

        self.direction = REVERSE
        self.index -= 1
        if self.index == -1:
            self.set_null()
        else:
            self.interval.left = breakpoints[self.index]
            self.interval.right = right
        return self.index != -1

    def seek_forward(self, index):
        # NOTE this is still in development and not fully tested.
        assert index >= self.index and index < self.ts.num_trees
        M = self.ts.num_edges
        breakpoints = self.ts.breakpoints(as_array=True)
        left_coords = self.ts.edges_left
        left_order = self.ts.indexes_edge_insertion_order
        right_coords = self.ts.edges_right
        right_order = self.ts.indexes_edge_removal_order

        if self.index == -1:
            self.interval.right = 0
            self.out_range.stop = 0
            self.in_range.stop = 0
            self.direction = FORWARD

        if self.direction == FORWARD:
            left_current_index = self.in_range.stop
            right_current_index = self.out_range.stop
        else:
            left_current_index = self.out_range.stop + 1
            right_current_index = self.in_range.stop + 1

        self.direction = FORWARD
        left = breakpoints[index]

        # The range of edges we need consider for removal starts
        # at the current right index and ends at the first edge
        # where the right coordinate is equal to the new tree's
        # left coordinate.
        j = right_current_index
        self.out_range.start = j
        # TODO This could be done with binary search
        while j < M and right_coords[right_order[j]] <= left:
            j += 1
        self.out_range.stop = j

        if self.index == -1:
            # No edges, so out_range should be empty
            self.out_range.start = self.out_range.stop

        # The range of edges we need to consider for the new tree
        # must have right coordinate > left
        j = left_current_index
        while j < M and right_coords[left_order[j]] <= left:
            j += 1
        self.in_range.start = j
        # TODO this could be done with a binary search
        while j < M and left_coords[left_order[j]] <= left:
            j += 1
        self.in_range.stop = j

        self.interval.left = left
        self.interval.right = breakpoints[index + 1]
        self.out_range.order = right_order
        self.in_range.order = left_order
        self.index = index

    def seek_backward(self, index):
        # NOTE this is still in development and not fully tested.
        assert index >= 0
        M = self.ts.num_edges
        breakpoints = self.ts.breakpoints(as_array=True)
        left_coords = self.ts.edges_left
        left_order = self.ts.indexes_edge_insertion_order
        right_coords = self.ts.edges_right
        right_order = self.ts.indexes_edge_removal_order

        if self.index == -1:
            assert index < self.ts.num_trees
            self.index = self.ts.num_trees
            self.interval.left = self.ts.sequence_length
            self.in_range.stop = M - 1
            self.out_range.stop = M - 1
            self.direction = REVERSE
        else:
            assert index <= self.index

        if self.direction == REVERSE:
            left_current_index = self.out_range.stop
            right_current_index = self.in_range.stop
        else:
            left_current_index = self.in_range.stop - 1
            right_current_index = self.out_range.stop - 1

        self.direction = REVERSE
        right = breakpoints[index + 1]

        # The range of edges we need consider for removal starts
        # at the current left index and ends at the first edge
        # where the left coordinate is equal to the new tree's
        # right coordinate.
        j = left_current_index
        self.out_range.start = j
        # TODO This could be done with binary search
        while j >= 0 and left_coords[left_order[j]] >= right:
            j -= 1
        self.out_range.stop = j

        if self.index == self.ts.num_trees:
            # No edges, so out_range should be empty
            self.out_range.start = self.out_range.stop

        # The range of edges we need to consider for the new tree
        # must have left coordinate < right
        j = right_current_index
        while j >= 0 and left_coords[right_order[j]] >= right:
            j -= 1
        self.in_range.start = j
        # We stop at the first edge with right coordinate < right
        while j >= 0 and right_coords[right_order[j]] >= right:
            j -= 1
        self.in_range.stop = j

        self.interval.right = right
        self.interval.left = breakpoints[index]
        self.out_range.order = left_order
        self.in_range.order = right_order
        self.index = index

    def step(self, direction):
        if direction == FORWARD:
            return self.next()
        elif direction == REVERSE:
            return self.prev()
        else:
            raise ValueError("Direction must be FORWARD (+1) or REVERSE (-1)")
