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


class TestCumsumOp(OpMapperTest):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
            self.place = paddle.CUDAPlace(0)
        else:
            self.target = DefaultHostTarget()
            self.place = paddle.CPUPlace()

    def init_input_data(self):
        self.shape = [2, 3, 4]
        self.axis = 1
        self.input_dtype = "float32"
        self.output_dtype = "float32"
        self.feed_data = {
            'x': self.random(self.shape, self.input_dtype),
        }

    def set_paddle_program(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype)
        out = paddle.cumsum(x, self.axis, self.output_dtype)

        return ([x.name], [out])

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestCumsumCase1(TestCumsumOp):
    """
    Test case with negative axis
    """

    def init_input_data(self):
        super().init_input_data()
        self.axis = -3


class TestCumsumCase2(TestCumsumOp):
    """
    Test case with unspecified axis and dtype
    """

    def init_input_data(self):
        super().init_input_data()
        self.axis = None
        self.output_dtype = None


class TestCumsumCase3(TestCumsumOp):
    """
    Test case with dtype int32
    """

    def init_input_data(self):
        self.shape = [2, 3, 4]
        self.axis = 1
        self.input_dtype = "int32"
        self.output_dtype = None
        self.feed_data = {
            'x': self.random(self.shape, self.input_dtype),
        }


class CumsumTestCase4(TestCumsumOp):
    """
    Test case with different input dtype and output dtype
    """

    def init_case(self):
        self.shape = [2, 3, 4]
        self.axis = 1
        self.input_dtype = "int32"
        self.output_dtype = "float32"
        self.feed_data = {
            'x': self.random(self.shape, self.input_dtype),
        }


if __name__ == "__main__":
    unittest.main()
