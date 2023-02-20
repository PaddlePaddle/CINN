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


class TestConv2dOp(OpMapperTest):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
            self.place = paddle.CUDAPlace(0)
        else:
            self.target = DefaultHostTarget()
            self.place = paddle.CPUPlace()

    def init_input_data(self):
        self.feed_data = {
            "x": self.random([3, 16, 32, 32], "float32"),
            "weight": self.random([16, 16, 3, 3], "float32")
        }
        self.data_format = 'NCHW'

    def set_paddle_program(self):
        x = paddle.static.data('x', self.feed_data["x"].shape,
                               self.feed_data["x"].dtype)
        weight = paddle.static.data('weight', self.feed_data["weight"].shape,
                                    self.feed_data["weight"].dtype)

        out = paddle.nn.functional.conv2d(
            x, weight, data_format=self.data_format)

        return ([x.name, weight.name], [out])

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestConv2dNCHWFP16(TestConv2dOp):
    def init_input_data(self):
        self.feed_data = {
            "x": self.random([3, 16, 32, 32], "float16"),
            "weight": self.random([16, 16, 3, 3], "float16")
        }
        self.data_format = 'NCHW'

    def test_check_results(self):
        self.check_outputs_and_grads(max_relative_error=1e-3)


class TestConv2dNHWC(TestConv2dOp):
    def init_input_data(self):
        self.feed_data = {
            "x": self.random([3, 32, 32, 16], "float32"),
            "weight": self.random([16, 16, 3, 3], "float32")
        }
        self.data_format = 'NHWC'


class TestConv2dNHWCFP16(TestConv2dOp):
    def init_input_data(self):
        self.feed_data = {
            "x": self.random([3, 32, 32, 16], "float16"),
            "weight": self.random([16, 16, 3, 3], "float16")
        }
        self.data_format = 'NHWC'

    def test_check_results(self):
        self.check_outputs_and_grads(max_relative_error=1e-3)


if __name__ == "__main__":
    unittest.main()
