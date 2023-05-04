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


class TestBroadcastToOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.case["x_shape"], dtype=self.case["x_dtype"])

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        out = paddle.broadcast_to(x, shape=self.case["d_shape"])

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("BroadcastTo")
        x = builder.create_input(
            self.nptype2cinntype(self.case["x_dtype"]), self.case["x_shape"],
            "x")
        out = builder.broadcast_to(
            x,
            out_shape=self.case["d_shape"],
            broadcast_axes=self.case["broadcast_axes"])

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.x_np], [out])

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
            },
            {
                "x_shape": [5, 3],
            },
            {
                "x_shape": [4, 3],
            },
            {
                "x_shape": [5],
            },
        ]
        self.dtypes = [
            {
                "x_dtype": "bool",
            },
            {
                "x_dtype": "int32",
            },
            {
                "x_dtype": "int64",
            },
            {
                "x_dtype": "float32",
            },
            {
                "x_dtype": "float64",
            },
        ]
        self.attrs = [
            {
                "d_shape": [4, 5, 3],
                "broadcast_axes": [0, 1, 2],
            },
            {
                "d_shape": [4, 5, 3],
                "broadcast_axes": [1, 2],
            },
            {
                "d_shape": [4, 5, 3],
                "broadcast_axes": [0, 2],
            },
            {
                "d_shape": [4, 5, 3],
                "broadcast_axes": [1],
            },
        ]


if __name__ == "__main__":
    TestBroadcastToAll().run()
