#!/usr/bin/env python3

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
class TestBinaryOp(OpTest):
    def setUp(self):
        self.init_case()

    def get_x_data(self):
        return self.random([32, 64])

    def get_y_data(self):
        return self.random([32, 64])

    def get_axis_value(self):
        return -1

    def init_case(self):
        self.inputs = {"x": self.get_x_data(), "y": self.get_y_data()}
        self.axis = self.get_axis_value()

    def paddle_func(self, x, y):
        return paddle.add(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.add(x, y, axis)

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=False)

        def get_unsqueeze_axis(x_rank, y_rank, axis):
            self.assertTrue(
                x_rank >= y_rank,
                "The rank of x should be greater or equal to that of y.")
            axis = axis if axis >= 0 else x_rank - y_rank
            unsqueeze_axis = np.arange(0, axis).tolist() + np.arange(
                axis + y_rank, x_rank).tolist()

            return unsqueeze_axis

        unsqueeze_axis = get_unsqueeze_axis(
            len(self.inputs["x"].shape), len(self.inputs["y"].shape),
            self.axis)
        y_t = paddle.unsqueeze(
            y, axis=unsqueeze_axis) if len(unsqueeze_axis) > 0 else y
        out = self.paddle_func(x, y_t)
        print("Paddle:", out)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("add")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        y = builder.create_input(Float(32), self.inputs["y"].shape, "y")
        out = self.cinn_func(builder, x, y, axis=self.axis)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x, y],
                                   [self.inputs["x"], self.inputs["y"]], [out])
        print("CINN:", res)
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()


test_op_list = [
    "add", "subtract", "divide", "multiply", "floor_divide", "mod",
    "floor_mod", "max", "min", "logical_and", "logical_or", "logical_xor",
    "bitwise_and", "bitwise_or", "bitwise_xor", "equal", "not_equal",
    "greater_than", "less_than", "greater_equal", "less_equal"
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
        "paddle_func":
        lambda _, x, y: eval(paddle_module_name + op_name)(x, y),
        "cinn_func":
        lambda _, builder, x, y, axis: eval("builder." + op_name.lower())(x, y,
                                                                          axis)
    }
    exec("test_class_" + op_name +
         " = type('Test' + op_name.title() + 'Op', (TestBinaryOp,), attrs)")

    case1_attrs = {"get_axis_value": lambda _: 0}
    exec(
        "test_case1_" + op_name +
        " = type('Test' + op_name.title() + 'Case1', (globals()['test_class_' + op_name],), case1_attrs)"
    )

if __name__ == "__main__":
    unittest.main()
