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
from op_test_helper import TestCaseHelper
import paddle
import cinn
from cinn.frontend import *
from cinn.common import *


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestUnaryOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.case["x_shape"], dtype=self.case["x_dtype"])

    def paddle_func(self, x):
        return paddle.abs(x)

    def cinn_func(self, builder, x):
        return builder.abs(x)

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        out = self.paddle_func(x)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("unary_elementwise_test")
        x = builder.create_input(
            self.nptype2cinntype(self.case["x_dtype"]), self.case["x_shape"],
            "x")
        out = self.cinn_func(builder, x)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.x_np], [out])

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestTrigon(TestUnaryOp):
    def paddle_func(self, x):
        op_type = self.case["op_type"]
        if op_type == "sin":
            return paddle.sin(x)
        elif op_type == "cos":
            return paddle.cos(x)
        elif op_type == "tan":
            return paddle.tan(x)
        elif op_type == "sinh":
            return paddle.sinh(x)
        elif op_type == "cosh":
            return paddle.cosh(x)
        elif op_type == "tanh":
            return paddle.tanh(x)

    def cinn_func(self, builder, x):
        op_type = self.case["op_type"]
        if op_type == "sin":
            return builder.sin(x)
        elif op_type == "cos":
            return builder.cos(x)
        elif op_type == "tan":
            return builder.tan(x)
        elif op_type == "sinh":
            return builder.sinh(x)
        elif op_type == "cosh":
            return builder.cosh(x)
        elif op_type == "tanh":
            return builder.tanh(x)


class TestUnaryTrigon(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestUnaryOpCase"
        self.cls = TestTrigon
        self.inputs = [
            {
                "x_shape": [1],
            },
            {
                "x_shape": [1024],
            },
            {
                "x_shape": [32, 64],
            },
            {
                "x_shape": [16, 8, 4, 2],
            },
        ]
        self.dtypes = [
            {
                "x_dtype": "float16",
                "max_relative_error": 1e-3
            },
            {
                "x_dtype": "float32",
            },
            {
                "x_dtype": "float64",
            },
        ]
        self.attrs = [{
            "op_type": "sin"
        }, {
            "op_type": "cos"
        }, {
            "op_type": "tan"
        }, {
            "op_type": "sinh"
        }, {
            "op_type": "cosh"
        }, {
            "op_type": "tanh"
        }]


class TestArcTrigon(TestUnaryOp):
    def prepare_inputs(self):
        if self.case["op_type"] == 'acosh':
            self.x_np = self.random(
                shape=self.case["x_shape"],
                dtype=self.case["x_dtype"],
                low=1.0,
                high=100.0)
        else:
            self.x_np = self.random(
                shape=self.case["x_shape"],
                dtype=self.case["x_dtype"],
                low=-1.0,
                high=1.0)

    def paddle_func(self, x):
        op_type = self.case["op_type"]
        if op_type == "asin":
            return paddle.asin(x)
        elif op_type == "acos":
            return paddle.acos(x)
        elif op_type == "atan":
            return paddle.atan(x)
        elif op_type == "asinh":
            return paddle.asinh(x)
        elif op_type == "acosh":
            return paddle.acosh(x)
        elif op_type == "atanh":
            return paddle.tanh(x)

    def cinn_func(self, builder, x):
        op_type = self.case["op_type"]
        if op_type == "asin":
            return builder.asin(x)
        elif op_type == "acos":
            return builder.acos(x)
        elif op_type == "atan":
            return builder.atan(x)
        elif op_type == "asinh":
            return builder.asinh(x)
        elif op_type == "acosh":
            return builder.acosh(x)
        elif op_type == "atanh":
            return builder.atanh(x)


class TestUnaryArcTrigon(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestUnaryOpCase"
        self.cls = TestArcTrigon
        self.inputs = [
            {
                "x_shape": [1],
            },
            {
                "x_shape": [1024],
            },
            {
                "x_shape": [32, 64],
            },
            {
                "x_shape": [16, 8, 4, 2],
            },
        ]
        self.dtypes = [
            {
                "x_dtype": "float16",
                "max_relative_error": 1e-3
            },
            {
                "x_dtype": "float32",
            },
            {
                "x_dtype": "float64",
            },
        ]
        self.attrs = [{
            "op_type": "asin"
        }, {
            "op_type": "acos"
        }, {
            "op_type": "atan"
        }, {
            "op_type": "asinh"
        }, {
            "op_type": "acosh"
        }, {
            "op_type": "atanh"
        }]


# class TestSqrtOp(TestUnaryOp):
#     def init_case(self):
#         self.inputs = {"x": self.random([32, 64], 'float32', 1.0, 1000.0)}

#     def paddle_func(self, x):
#         return paddle.sqrt(x)

#     def cinn_func(self, builder, x):
#         return builder.sqrt(x)

# class TestSqrtOpFP64(TestSqrtOp):
#     def init_case(self):
#         self.inputs = {"x": self.random([32, 64], 'float64', 1.0, 1000.0)}

# class TestReluOp(TestUnaryOp):
#     def paddle_func(self, x):
#         return paddle.nn.functional.relu(x)

#     def cinn_func(self, builder, x):
#         return builder.relu(x)

# class TestSigmoidOp(TestUnaryOp):
#     def paddle_func(self, x):
#         return paddle.nn.functional.sigmoid(x)

#     def cinn_func(self, builder, x):
#         return builder.sigmoid(x)

# class TestIdentityOp(TestUnaryOp):
#     def paddle_func(self, x):
#         return paddle.assign(x)

#     def cinn_func(self, builder, x):
#         return builder.identity(x)

# class TestExpOp(TestUnaryOp):
#     def paddle_func(self, x):
#         return paddle.exp(x)

#     def cinn_func(self, builder, x):
#         return builder.exp(x)

# class TestExpOpFP64(TestExpOp):
#     def init_case(self):
#         self.inputs = {"x": self.random([32, 64], 'float64', -10.0, 10.0)}

# class TestErfOp(TestUnaryOp):
#     def paddle_func(self, x):
#         return paddle.erf(x)

#     def cinn_func(self, builder, x):
#         return builder.erf(x)

# class TestRsqrtOp(TestUnaryOp):
#     def init_case(self):
#         self.inputs = {"x": self.random([32, 64], 'float32', 0.00001, 1.0)}

#     def paddle_func(self, x):
#         return paddle.rsqrt(x)

#     def cinn_func(self, builder, x):
#         return builder.rsqrt(x)

# class TestLogOp(TestUnaryOp):
#     def init_case(self):
#         self.inputs = {"x": self.random([32, 64], 'float32', 1.0, 10.0)}

#     def paddle_func(self, x):
#         return paddle.log(x)

#     def cinn_func(self, builder, x):
#         return builder.log(x)

# class TestLog2Op(TestUnaryOp):
#     def init_case(self):
#         self.inputs = {"x": self.random([32, 64], 'float32', 1.0, 10.0)}

#     def paddle_func(self, x):
#         return paddle.log2(x)

#     def cinn_func(self, builder, x):
#         return builder.log2(x)

# class TestLog10Op(TestUnaryOp):
#     def init_case(self):
#         self.inputs = {"x": self.random([32, 64], 'float32', 1.0, 10.0)}

#     def paddle_func(self, x):
#         return paddle.log10(x)

#     def cinn_func(self, builder, x):
#         return builder.log10(x)

# class TestFloorOp(TestUnaryOp):
#     def paddle_func(self, x):
#         return paddle.floor(x)

#     def cinn_func(self, builder, x):
#         return builder.floor(x)

# class TestCeilOp(TestUnaryOp):
#     def paddle_func(self, x):
#         return paddle.ceil(x)

#     def cinn_func(self, builder, x):
#         return builder.ceil(x)

# class TestRoundOp(TestUnaryOp):
#     def paddle_func(self, x):
#         return paddle.round(x)

#     def cinn_func(self, builder, x):
#         return builder.round(x)

# class TestTruncOp(TestUnaryOp):
#     def paddle_func(self, x):
#         return paddle.trunc(x)

#     def cinn_func(self, builder, x):
#         return builder.trunc(x)

# class TestLogicalNotOp(TestUnaryOp):
#     def init_case(self):
#         self.inputs = {"x": self.random([32, 64], 'bool')}

#     def paddle_func(self, x):
#         return paddle.logical_not(x)

#     def cinn_func(self, builder, x):
#         return builder.logical_not(x)

# class TestBitwiseNotOp(TestUnaryOp):
#     def init_case(self):
#         self.inputs = {"x": self.random([32, 64], 'int32', 1, 10000)}

#     def paddle_func(self, x):
#         return paddle.bitwise_not(x)

#     def cinn_func(self, builder, x):
#         return builder.bitwise_not(x)

# class TestSignOp(TestUnaryOp):
#     def paddle_func(self, x):
#         return paddle.sign(x)

#     def cinn_func(self, builder, x):
#         return builder.sign(x)

# class TestAbsOp(TestUnaryOp):
#     def paddle_func(self, x):
#         return paddle.abs(x)

#     def cinn_func(self, builder, x):
#         return builder.abs(x)

# class TestIsNanOp(TestUnaryOp):
#     def paddle_func(self, x):
#         return paddle.isnan(x)

#     def cinn_func(self, builder, x):
#         return builder.is_nan(x)

# class TestIsNanCase1(TestIsNanOp):
#     def init_case(self):
#         self.inputs = {"x": self.random([32, 64])}
#         self.inputs["x"][0] = [np.nan] * 64

# class TestIsFiniteOp(TestUnaryOp):
#     def paddle_func(self, x):
#         return paddle.isfinite(x)

#     def cinn_func(self, builder, x):
#         return builder.is_finite(x)

# class TestIsFiniteCase1(TestIsFiniteOp):
#     def init_case(self):
#         self.inputs = {"x": self.random([32, 64])}
#         self.inputs["x"][0] = [np.inf] * 64

# class TestIsInfOp(TestUnaryOp):
#     def paddle_func(self, x):
#         return paddle.isinf(x)

#     def cinn_func(self, builder, x):
#         return builder.is_inf(x)

# class TestIsInfCase1(TestIsInfOp):
#     def init_case(self):
#         self.inputs = {"x": self.random([32, 64])}
#         self.inputs["x"][0] = [np.inf] * 64

# class TestNegOp(TestUnaryOp):
#     def paddle_func(self, x):
#         return paddle.neg(x)

#     def cinn_func(self, builder, x):
#         return builder.negative(x)

# class TestNegCase1(TestNegOp):
#     def init_case(self):
#         self.inputs = {"x": self.random([32, 64], low=-1.0, high=1.0)}

if __name__ == "__main__":
    TestUnaryTrigon().run()
    TestUnaryArcTrigon().run()
