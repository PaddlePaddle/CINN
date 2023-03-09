#!/usr/bin/env python3

# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

import sys
import unittest
import numpy as np
from op_mapper_test import OpMapperTest
import paddle
from cinn.frontend import *
from cinn.common import *

paddle.enable_static()

enable_gpu = sys.argv.pop()


class TestRollOp(OpMapperTest):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
            self.place = paddle.CUDAPlace(0)
        else:
            self.target = DefaultHostTarget()
            self.place = paddle.CPUPlace()

    def init_input_data(self):
        self.feed_data = {
            'x': np.array([1, 2, 3], dtype='float32'),
        }
        self.axis = None
        self.shifts = 1

    def set_paddle_program(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype)

        out = paddle.roll(x, self.shifts, self.axis)

        return ([x.name], [out])

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestRollCase1(TestTileOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([2, 3, 4], 'float32'),
        }
        self.axis = 0
        self.shifts = 2


class TestRollCase2(TestTileOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([2, 10, 5], 'float32'),
        }
        self.axis = [1]
        self.shifts = [3]


class TestRollCase3(TestTileOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([2, 4, 15], 'float32'),
        }
        self.axis = [0, 1, 2]
        self.shifts = [3, 4, 10]


class TestRollCase4(TestTileOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([12, 14], 'float32'),
        }
        self.axis = [0, 1]
        self.shifts = [3, -8]


class TestRollCase5(TestTileOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([120], 'float32'),
        }
        self.axis = [0]
        self.shifts = [121]


if __name__ == "__main__":
    unittest.main()
