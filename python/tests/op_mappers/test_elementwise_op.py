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


class TestElementwiseOp(OpMapperTest):
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
            'y': self.random([32, 64], "float32")
        }

    def set_elementwise_func(self, x, y):
        return paddle.add(x, y)

    def set_paddle_program(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype)
        y = paddle.static.data(
            name='x',
            shape=self.feed_data['y'].shape,
            dtype=self.feed_data['y'].dtype)
        out = self.set_elementwise_func(x, y)

        return ([x.name, y.name], [out])

    def test_check_results(self):
        self.check_outputs_and_grads()


test_op_list = [
    'add', 'subtract', 'divide', 'multiply', 'pow', 'mod', 'maximum', 'minimum'
]

for op_name in test_op_list:
    paddle_module_name = ""
    if hasattr(paddle, op_name):
        paddle_module_name = "paddle."
    elif hasattr(paddle.nn, op_name):
        paddle_module_name = "paddle.nn."
    elif hasattr(paddle.nn.functional, op_name):
        paddle_module_name = "paddle.nn.functional."
    else:
        assert False, op_name + " should in 'paddle' or 'paddle.nn.functional' module!"

    attrs = {
        "set_elementwise_func":
        lambda _, x: eval(paddle_module_name + op_name)(x, y)
    }
    exec(
        "test_class_" + op_name +
        " = type('Test' + op_name.title() + 'Op', (TestElementwiseOp,), attrs)"
    )
