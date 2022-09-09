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

import unittest
import numpy as np
from op_test import OpTest, OpTestTool
import paddle
import cinn
from cinn.frontend import *
from cinn.common import *


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestUnaryOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {"x": self.random([32, 64])}

    def paddle_func(self, x):
        return paddle.sqrt(x)

    def cinn_func(self, builder, x):
        return builder.sqrt(x)

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        out = self.paddle_func(x)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("sigmoid")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        out = self.cinn_func(builder, x)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]],
                                   [out])

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()


test_op_list = [
    'sqrt', 'relu', 'sigmoid', 'Identity', 'exp', 'erf', 'rsqrt', 'log',
    'log2', 'log10', 'floor', 'ceil', 'round', 'trunc', 'sin', 'cos', 'tan',
    'sinh', 'cosh', 'tanh', 'asin', 'acos', 'atan', 'asinh', 'acosh', 'atanh',
    'logical_not', 'bitwise_not', 'sign', 'abs'
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
        "paddle_func": lambda _, x: eval(paddle_module_name + op_name)(x),
        "cinn_func": lambda _, builder, x: eval("builder." + op_name.lower())(x
                                                                              )
    }
    exec("test_class_" + op_name +
         " = type('Test' + op_name.title() + 'Op', (TestUnaryOp,), attrs)")


class TestIsNanOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.isnan(x)

    def cinn_func(self, builder, x):
        return builder.is_nan(x)


class TestIsNanCase1(TestIsNanOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64])}
        self.inputs["x"][0] = [np.nan] * 64


class TestIsFiniteOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.isfinite(x)

    def cinn_func(self, builder, x):
        return builder.is_finite(x)


class TestIsFiniteCase1(TestIsFiniteOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64])}
        self.inputs["x"][0] = [np.inf] * 64


class TestIsInfOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.isinf(x)

    def cinn_func(self, builder, x):
        return builder.is_inf(x)


class TestIsInfCase1(TestIsInfOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64])}
        self.inputs["x"][0] = [np.inf] * 64


class TestNegOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.neg(x)

    def cinn_func(self, builder, x):
        return builder.negative(x)


class TestNegCase1(TestNegOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64], low=-1.0, high=1.0)}


if __name__ == "__main__":
    unittest.main()
