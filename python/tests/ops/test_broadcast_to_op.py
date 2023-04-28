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
from op_test_helper import TestCaseHelper
import paddle
import cinn
from cinn.frontend import *
from cinn.common import *
import sys



@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestBroadcastToOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x = self.random(
            shape=self.case["x_shape"],
            dtype=self.case["x_dtype"],
            low=-10,
            high=10
        )
        self.out = self.random(
            shape=self.case["out_shape"],
            dtype=self.case["out_dtype"],
            low=-10,
            high=10
        )
        self.broadcast_axes = self.random(
            shape=self.case["axes_shape"],
            dtype=self.case["axes_dtype"],
            low=-10,
            high=10
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x, stop_gradient=True)
        out = paddle.broadcast_to(x, shape=self.out)

        self.paddle_outputs = [out]

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("BroadcastTo")
        x = builder.create_input(self.nptype2cinntype(self.case["x_dtype"]), self.case["x_shape"],
            "x")
        out = builder.broadcast_to(
            x, out_shape=self.out, broadcast_axes=self.broadcast_axes)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.x],
                                   [out])

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        max_relative_error = self.case[
            "max_relative_error"] if "max_relative_error" in self.case else 1e-5
        self.check_outputs_and_grads(max_relative_error=max_relative_error)


class TestBroadcastToAll(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestBroadcastToOpCase"
        self.cls = TestBroadcastToOp
        self.inputs = [
            {
                "x_shape": [1, 1, 3],
                "out_shape": [4, 5, 3],
                "axes_shape": [0, 1, 2],
            },
            {
                "x_shape": [5, 3],
                "out_shape": [4, 5, 3],
                "axes_shape": [1, 2],
            },
        ]
        self.dtypes = [
            {
                "x_dtype": "int16",
                "out_dtype": "int16",
                "axes_dtype": "int16",
            },
            {
                "x_dtype": "int32",
                "out_dtype": "int32",
                "axes_dtype": "int32",
            },
            {
                "x_dtype": "int64",
                "out_dtype": "int64",
                "axes_dtype": "int64",
            },
            {
                "x_dtype": "float16",
                "out_dtype": "float16",
                "axes_dtype": "int32",
                "max_relative_error": 1e-3,
            },
            {
                "x_dtype": "float32",
                "out_dtype": "float32",
                "axes_dtype": "int32",
            },
            {
                "x_dtype": "float64",
                "out_dtype": "float64",
                "axes_dtype": "int32",
            },
        ]
        self.attrs = []


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestBroadcastToNoAxesOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x = self.random(
            shape=self.case["x_shape"],
            dtype=self.case["x_dtype"],
            low=-10,
            high=10
        )
        self.out = self.random(
            shape=self.case["out_shape"],
            dtype=self.case["out_dtype"],
            low=-10,
            high=10
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x, stop_gradient=True)
        out = paddle.broadcast_to(x, shape=self.out)

        self.paddle_outputs = [out]

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("BroadcastTo")
        x = builder.create_input(self.nptype2cinntype(self.case["x_dtype"]), self.case["x_shape"],
            "x")
        out = builder.broadcast_to(
            x, out_shape=self.out, broadcast_axes=self.broadcast_axes)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.x],
                                   [out])

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        max_relative_error = self.case[
            "max_relative_error"] if "max_relative_error" in self.case else 1e-5
        self.check_outputs_and_grads(max_relative_error=max_relative_error)


class TestBroadcastToNoAxesAll(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestBroadcastToNoAxesOpCase"
        self.cls = TestBroadcastToNoAxesOp
        self.inputs = [
            {
                "x_shape": [6],
                "out_shape": [4, 5, 6],
                "axes_shape": [],
            },
            {
                "x_shape": [1, 1, 3],
                "out_shape": [4, 5, 3],
                "axes_shape": [],
            },
            {
                "x_shape": [5, 3],
                "out_shape": [4, 5, 3],
                "axes_shape": [],
            },
            {
                "x_shape": [4, 1, 3],
                "out_shape": [4, 5, 3],
                "axes_shape": [],
            },
            {
                "x_shape": [1, 1, 1],
                "out_shape": [4, 5, 3],
                "axes_shape": [],
            },
            {
                "x_shape": [1],
                "out_shape": [5],
                "axes_shape": [],
            },
        ]
        self.dtypes = [
            {
                "x_dtype": "int16",
                "out_dtype": "int16",
                "axes_dtype": "int16",
            },
            {
                "x_dtype": "int32",
                "out_dtype": "int32",
                "axes_dtype": "int32",
            },
            {
                "x_dtype": "int64",
                "out_dtype": "int64",
                "axes_dtype": "int64",
            },
            {
                "x_dtype": "float16",
                "out_dtype": "float16",
                "axes_dtype": "int32",
                "max_relative_error": 1e-3,
            },
            {
                "x_dtype": "float32",
                "out_dtype": "float32",
                "axes_dtype": "int32",
            },
            {
                "x_dtype": "float64",
                "out_dtype": "float64",
                "axes_dtype": "int32",
            },
        ]
        self.attrs = []


if __name__ == "__main__":
    TestBroadcastToAll().run()
    TestBroadcastToNoAxesAll().run()