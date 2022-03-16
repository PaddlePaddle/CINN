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
class TestConcatOp1(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x1": np.random.random([2, 3]).astype("float32"),
            "x2": np.random.random((2, 3)).astype("float32")
        }
        self.axis = 0

    def build_paddle_program(self, target):
        x1 = paddle.to_tensor(self.inputs["x1"], stop_gradient=True)
        x2 = paddle.to_tensor(self.inputs["x2"], stop_gradient=True)
        out = paddle.concat(x=[x1, x2], axis=self.axis)

        self.paddle_outputs = [out]

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("concat")
        x1 = builder.create_input(Float(32), self.inputs["x1"].shape, "x1")
        x2 = builder.create_input(Float(32), self.inputs["x2"].shape, "x2")
        out = builder.concat([x1, x2], axis=self.axis)

        prog = builder.build()
        res = self.get_cinn_output(prog, set(), target, [x1, x2],
                                   [self.inputs["x1"], self.inputs["x2"]],
                                   [out])

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestConcat2Case1(TestConcatOp1):
    def init_case(self):
        self.inputs = {
            "x1": np.random.random([4, 3]).astype("float32"),
            "x2": np.random.random((8, 3)).astype("float32")
        }
        self.axis = 0


class TestConcat2Case2(TestConcatOp1):
    def init_case(self):
        self.inputs = {
            "x1": np.random.random([2, 4, 8]).astype("float32"),
            "x2": np.random.random((2, 4, 4)).astype("float32")
        }
        self.axis = -1


class TestConcat2Case3(TestConcatOp1):
    def init_case(self):
        self.inputs = {
            "x1": np.random.random([2, 8, 4]).astype("float32"),
            "x2": np.random.random((2, 4, 4)).astype("float32")
        }
        self.axis = 1


class TestConcatOp2(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x1": np.random.random([1, 16]).astype("float32"),
            "x2": np.random.random([2, 16]).astype("float32"),
            "x3": np.random.random([3, 16]).astype("float32"),
            "x4": np.random.random([4, 16]).astype("float32"),
            "x5": np.random.random([5, 16]).astype("float32")
        }
        self.axis = 0

    def build_paddle_program(self, target):
        x1 = paddle.to_tensor(self.inputs["x1"], stop_gradient=True)
        x2 = paddle.to_tensor(self.inputs["x2"], stop_gradient=True)
        x3 = paddle.to_tensor(self.inputs["x3"], stop_gradient=True)
        x4 = paddle.to_tensor(self.inputs["x4"], stop_gradient=True)
        x5 = paddle.to_tensor(self.inputs["x5"], stop_gradient=True)
        out = paddle.concat(x=[x1, x2, x3, x4, x5], axis=self.axis)

        self.paddle_outputs = [out]

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("concat")
        x1 = builder.create_input(Float(32), self.inputs["x1"].shape, "x1")
        x2 = builder.create_input(Float(32), self.inputs["x2"].shape, "x2")
        x3 = builder.create_input(Float(32), self.inputs["x3"].shape, "x3")
        x4 = builder.create_input(Float(32), self.inputs["x4"].shape, "x4")
        x5 = builder.create_input(Float(32), self.inputs["x5"].shape, "x5")
        out = builder.concat([x1, x2, x3, x4, x5], axis=self.axis)

        prog = builder.build()
        res = self.get_cinn_output(prog, set(), target, [x1, x2, x3, x4, x5], [
            self.inputs["x1"], self.inputs["x2"], self.inputs["x3"],
            self.inputs["x4"], self.inputs["x5"]
        ], [out])

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
