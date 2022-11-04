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


class TestMulOp(OpMapperTest):
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
            'y': self.random([64, 32], "float32"),
        }
        self.x_num_col_dims = 1
        self.y_num_col_dims = 1

    def set_paddle_program(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype)
        y = paddle.static.data(
            name='y',
            shape=self.feed_data['y'].shape,
            dtype=self.feed_data['x'].dtype)
        out = paddle.fluid.layers.mul(x, y, self.x_num_col_dims,
                                      self.y_num_col_dims)

        return ([x.name, y.name], [out])

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestMulCase1(TestMulOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([4, 8, 64], "float32"),
            'y': self.random([64, 4, 8], "float32"),
        }
        self.x_num_col_dims = 2
        self.y_num_col_dims = 1


class TestMulCase2(TestMulOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([4, 8, 8, 8], "float32"),
            'y': self.random([8, 8, 4, 8], "float32"),
        }
        self.x_num_col_dims = 2
        self.y_num_col_dims = 2


if __name__ == "__main__":
    unittest.main()
