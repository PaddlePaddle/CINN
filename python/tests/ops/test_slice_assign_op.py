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
class TestSliceAssignOp(OpTest):
    def setUp(self):
        self.init_case()
        self.prepare_case()

    def init_case(self):
        self.inputs = {
            "inputs": np.random.random([10, 12]).astype("float32"),
            "assign": np.zeros([3, 3]).astype("float32")
        }
        self.axes = [0, 1]
        self.starts = [2, 2]
        self.ends = [5, 5]
        self.strides = [1, 1]

    def prepare_case(self):
        for i in range(len(self.ends)):
            input_len = self.inputs["inputs"].shape[i]
            if self.ends[i] < 0:
                self.ends[i] += input_len
            elif self.ends[i] > input_len:
                self.ends[i] = input_len
            if self.starts[i] < 0:
                self.starts[i] += input_len
            elif self.starts[i] > input_len:
                self.starts[i] = input_len - 1

    def num_of_slice(self, start, end, stride):
        if stride < 0:
            start, end = end, start
            stride = -stride
        num = 0
        while start < end:
            start += stride
            num += 1
        return num

    def build_paddle_program(self, target):
        res = self.inputs["inputs"].copy()

        row_len = self.num_of_slice(self.starts[0], self.ends[0],
                                    self.strides[0])

        for row_id in range(row_len):
            res[self.starts[0] + self.strides[0] *
                row_id][self.starts[1]:self.ends[1]:self.
                        strides[1]] = self.inputs["assign"][row_id]

        pd_res = paddle.to_tensor(res, stop_gradient=True)
        self.paddle_outputs = [pd_res]

    def build_cinn_program(self, target):
        builder = NetBuilder("slice_assign")
        inputs = builder.create_input(
            Float(32), self.inputs["inputs"].shape, "inputs")
        assign = builder.create_input(
            Float(32), self.inputs["assign"].shape, "assign")
        out = builder.slice_assign(
            inputs,
            assign,
            starts=self.starts,
            ends=self.ends,
            axes=self.axes,
            strides=self.strides)

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [inputs, assign],
            [self.inputs["inputs"], self.inputs["assign"]], [out])
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestSliceAssignCase1(TestSliceAssignOp):
    def init_case(self):
        self.inputs = {
            "inputs": np.random.random([10, 12]).astype("float32"),
            "assign": np.zeros([5, 5]).astype("float32")
        }
        self.axes = [0, 1]
        self.starts = [1, 2]
        self.ends = [6, 1000]
        self.strides = [1, 2]


class TestSliceAssignCase2(TestSliceAssignOp):
    def init_case(self):
        self.inputs = {
            "inputs": np.random.random([10, 12]).astype("float32"),
            "assign": np.zeros([3, 3]).astype("float32")
        }
        self.axes = [0, 1]
        self.starts = [2, 1]
        self.ends = [-1, 7]
        self.strides = [3, 2]


class TestSliceAssignCase3(TestSliceAssignOp):
    def init_case(self):
        self.inputs = {
            "inputs": np.random.random([10, 12]).astype("float32"),
            "assign": np.zeros([7, 5]).astype("float32")
        }
        self.axes = [0, 1]
        self.starts = [2, 1000]
        self.ends = [8, 1]
        self.strides = [1, -2]


class TestSliceAssignCase4(TestSliceAssignOp):
    def init_case(self):
        self.inputs = {
            "inputs": np.random.random([10, 12]).astype("float32"),
            "assign": np.zeros([4, 3]).astype("float32")
        }
        self.axes = [0, 1]
        self.starts = [-1, -2]
        self.ends = [-5, -8]
        self.strides = [-1, -2]


class TestSliceAssignAxes1Op(TestSliceAssignOp):
    def init_case(self):
        self.inputs = {
            "inputs": np.random.random([121, 2]).astype("float32"),
            "assign": np.zeros([121, 1]).astype("float32")
        }
        self.axes = [1]
        self.starts = [0]
        self.ends = [1]
        self.strides = [1]

    def build_paddle_program(self, target):
        res = self.inputs["inputs"].copy()

        for row_id in range(self.inputs["inputs"].shape[0]):
            res[row_id][self.starts[0]:self.ends[0]:self.
                        strides[0]] = self.inputs["assign"][row_id]

        pd_res = paddle.to_tensor(res, stop_gradient=True)
        self.paddle_outputs = [pd_res]


class TestSliceAssignAxes1Case1(TestSliceAssignAxes1Op):
    def init_case(self):
        self.inputs = {
            "inputs": np.random.random([121, 2]).astype("float32"),
            "assign": np.zeros([121, 1]).astype("float32")
        }
        self.axes = [1]
        self.starts = [1]
        self.ends = [2]
        self.strides = [1]


class TestSliceAssignAxes1Case2(TestSliceAssignAxes1Op):
    def init_case(self):
        self.inputs = {
            "inputs": np.random.random([121, 2]).astype("float32"),
            "assign": np.zeros([121, 1]).astype("float32")
        }
        self.axes = [1]
        self.starts = [1]
        self.ends = [0]
        self.strides = [-1]


class TestSliceAssignAxes2Op(TestSliceAssignOp):
    def init_case(self):
        self.inputs = {
            "inputs": np.random.random([121, 2, 2]).astype("float32"),
            "assign": np.zeros([121, 2, 1]).astype("float32")
        }
        self.axes = [2]
        self.starts = [0]
        self.ends = [1]
        self.strides = [1]

    def build_paddle_program(self, target):
        res = self.inputs["inputs"].copy()

        row_len = self.num_of_slice(self.starts[0], self.ends[0],
                                    self.strides[0])

        for row_id in range(self.inputs["inputs"].shape[0]):
            for col_id in range(self.inputs["inputs"].shape[1]):
                res[row_id][col_id][self.starts[0]:self.ends[0]:self.
                                    strides[0]] = self.inputs["assign"][
                                        row_id][col_id]

        pd_res = paddle.to_tensor(res, stop_gradient=True)
        self.paddle_outputs = [pd_res]


class TestSliceAssignAxes2Case1(TestSliceAssignAxes2Op):
    def init_case(self):
        self.inputs = {
            "inputs": np.random.random([121, 2, 2]).astype("float32"),
            "assign": np.zeros([121, 2, 1]).astype("float32")
        }
        self.axes = [2]
        self.starts = [1]
        self.ends = [2]
        self.strides = [1]


class TestSliceAssignAxes2Case2(TestSliceAssignAxes2Op):
    def init_case(self):
        self.inputs = {
            "inputs": np.random.random([121, 2, 2]).astype("float32"),
            "assign": np.zeros([121, 2, 1]).astype("float32")
        }
        self.axes = [2]
        self.starts = [1]
        self.ends = [0]
        self.strides = [-1]


if __name__ == "__main__":
    unittest.main()
