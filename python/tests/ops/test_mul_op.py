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

from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper
import paddle
from cinn.frontend import *
from cinn.common import *
import sys


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestMulOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.case["x_shape"],
            dtype=self.case["x_dtype"],
            low=self.case["x_low"],
            high=self.case["x_high"])
        self.y_np = self.random(
            shape=self.case["y_shape"],
            dtype=self.case["y_dtype"],
            low=self.case["y_low"],
            high=self.case["y_high"])

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        y = paddle.to_tensor(self.y_np, stop_gradient=True)
        out = paddle.matmul(x, y)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("matmul")
        x = builder.create_input(
            self.nptype2cinntype(self.case["x_dtype"]), self.case["x_shape"],
            "x")
        y = builder.create_input(
            self.nptype2cinntype(self.case["y_dtype"]), self.case["y_shape"],
            "y")
        out = builder.matmul(x, y)
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x, y],
                                   [self.x_np, self.y_np], [out])

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        max_relative_error = self.case[
            "max_relative_error"] if "max_relative_error" in self.case else 1e-5
        self.check_outputs_and_grads(max_relative_error=max_relative_error)


class TestMulOpBase(TestCaseHelper):
    inputs = [{
        "x_shape": [1],
        "y_shape": [1],
    }, {
        "x_shape": [1024],
        "y_shape": [1024],
    }, {
        "x_shape": [32, 64],
        "y_shape": [64, 32],
    }, {
        "x_shape": [2, 3, 4],
        "y_shape": [2, 4, 3],
    }, {
        "x_shape": [16, 8, 4, 2],
        "y_shape": [16, 8, 2, 4],
    }]

    dtypes = [
        {
            "x_dtype": "float32",
            "y_dtype": "float32",
        },
    ]

    attrs = [
        {
            "x_low": -10,
            "x_high": 10,
            "y_low": -10,
            "y_high": 10,
        },
    ]

    def init_attrs(self):
        self.class_name = "TestMulOpBase"
        self.cls = TestMulOp


class TestMulOpShapeTest(TestMulOpBase):
    def init_attrs(self):
        self.class_name = "TestMulOpShapeTest"
        self.cls = TestMulOp
        self.inputs = [{
            "x_shape": [1],
            "y_shape": [1],
        }, {
            "x_shape": [1024],
            "y_shape": [1024],
        }, {
            "x_shape": [2048],
            "y_shape": [2048],
        }, {
            "x_shape": [32, 64],
            "y_shape": [64, 32],
        }, {
            "x_shape": [2, 3, 4],
            "y_shape": [2, 4, 3],
        }, {
            "x_shape": [16, 8, 4, 2],
            "y_shape": [16, 8, 2, 4],
        }]


class TestMulOpDtypeTest(TestMulOpBase):
    def init_attrs(self):
        self.class_name = "TestMulOpDtypeTest"
        self.cls = TestMulOp
        self.dtypes = [
            #{
            #"x_dtype": "float16",
            #"y_dtype": "float16",
            #"max_relative_error": 1e-3,
        #}, 
        {
            "x_dtype": "float32",
            "y_dtype": "float32",
        }, {
            "x_dtype": "float64",
            "y_dtype": "float64",
        }]


class TestMulOpPolarityTest(TestMulOpBase):
    def init_attrs(self):
        self.class_name = "TestMulOpPolarityTest"
        self.cls = TestMulOp
        self.attrs = [
            {
                "x_low": -10,
                "x_high": 10,
                "y_low": -10,
                "y_high": 10,
            },
        ]


class TestMulOpBroadcastTest(TestMulOpBase):
    def init_attrs(self):
        self.class_name = "TestMulOpBroadcastTest"
        self.cls = TestMulOp
        self.inputs = [{
            "x_shape": [1],
            "y_shape": [1, 1],
        }, {
            "x_shape": [1, 64],
            "y_shape": [1, 64, 1],
        }, {
            "x_shape": [1, 3, 4],
            "y_shape": [1, 3, 4, 2],
        }, {
            "x_shape": [12, 1, 4, 2],
            "y_shape": [12, 1, 2, 4],
        }]


if __name__ == "__main__":
    TestMulOpShapeTest().run()
    TestMulOpDtypeTest().run()
    TestMulOpPolarityTest().run()
    TestMulOpBroadcastTest().run()
