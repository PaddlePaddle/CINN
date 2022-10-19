#!/usr/bin/env python3

# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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


class TestSplitOp(OpMapperTest):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
            self.place = paddle.CUDAPlace(0)
        else:
            self.target = DefaultHostTarget()
            self.place = paddle.CPUPlace()

    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "float32"),
        }
        self.axis = 0
        self.num_or_sections = 4

    def set_paddle_program(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype)
        out = paddle.split(x, self.num_or_sections, self.axis)

        return ([x.name], out)

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestSplitCase1(TestSplitOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "float32"),
        }
        self.axis = 0
        self.num_or_sections = [3, 5, 16, 2, 6]


class TestSplitCase2(TestSplitOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "float32"),
        }
        self.axis = -1
        self.num_or_sections = 8


class TestSplitCase3(TestSplitOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "int32", 0, 10000),
        }
        self.axis = 0
        self.num_or_sections = [4, 4, 16, 8]


if __name__ == "__main__":
    unittest.main()
