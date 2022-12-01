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
import paddle.nn.functional as F
import cinn
from cinn.frontend import *
from cinn.common import *


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestMatmulOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x": np.random.random([4, 16]).astype("float32"),
            "y": np.random.random([16, 32]).astype("float32")
        }
        self.transpose_x = False
        self.transpose_y = False

    def paddle_func(self, x, y):
        return paddle.matmul(
            x, y, transpose_x=self.transpose_x, transpose_y=self.transpose_y)

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=True)

        out = self.paddle_func(x, y)

        self.paddle_outputs = [out]

    def cinn_func(self, builder, x, y):
        return builder.matmul(
            x, y, transpose_x=self.transpose_x, transpose_y=self.transpose_y)

    def build_cinn_program(self, target):
        builder = NetBuilder("matmul")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        y = builder.create_input(Float(32), self.inputs["y"].shape, "y")
        out = self.cinn_func(builder, x, y)

        prog = builder.build()
        res = self.get_cinn_output(
            prog,
            target, [x, y], [self.inputs["x"], self.inputs["y"]], [out],
            passes=list())

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestMatmulCase1(TestMatmulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([2, 16]).astype("float32"),
            "y": np.random.random([16, 2]).astype("float32")
        }
        self.transpose_x = False
        self.transpose_y = False


class TestMatmulCase2(TestMatmulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([5, 4, 16]).astype("float32"),
            "y": np.random.random([5, 16, 32]).astype("float32")
        }
        self.transpose_x = False
        self.transpose_y = False


class TestMatmulCase3(TestMatmulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([5, 16, 4]).astype("float32"),
            "y": np.random.random([5, 16, 32]).astype("float32")
        }
        self.transpose_x = True
        self.transpose_y = False


class TestMatmulCase4(TestMatmulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([16, 4]).astype("float32"),
            "y": np.random.random([16, 32]).astype("float32")
        }
        self.transpose_x = True
        self.transpose_y = False


class TestMatmulCase5(TestMatmulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([4, 16]).astype("float32"),
            "y": np.random.random([32, 16]).astype("float32")
        }
        self.transpose_x = False
        self.transpose_y = True


class TestMatmulCase6(TestMatmulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([16, 4]).astype("float32"),
            "y": np.random.random([32, 16]).astype("float32")
        }
        self.transpose_x = True
        self.transpose_y = True


class TestMatmulCase7(TestMatmulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([8, 16, 4]).astype("float32"),
            "y": np.random.random([1, 4, 16]).astype("float32")
        }
        self.transpose_x = False
        self.transpose_y = False


class TestMatmulCase8(TestMatmulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([8, 16, 4]).astype("float32"),
            "y": np.random.random([1, 4, 16]).astype("float32")
        }
        self.transpose_x = False
        self.transpose_y = False


class TestMatmulCase9(TestMatmulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([8, 16, 4]).astype("float32"),
            "y": np.random.random([4, 16]).astype("float32")
        }
        self.transpose_x = False
        self.transpose_y = False


class TestMatmulCase10(TestMatmulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([8, 16, 4]).astype("float32"),
            "y": np.random.random([4, 16]).astype("float32")
        }
        self.transpose_x = False
        self.transpose_y = False


class TestMatmulCase11(TestMatmulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([8, 4, 16]).astype("float32"),
            "y": np.random.random([4, 16]).astype("float32")
        }
        self.transpose_x = True
        self.transpose_y = False


class TestMatmulCase12(TestMatmulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([8, 16, 4]).astype("float32"),
            "y": np.random.random([16, 1]).astype("float32")
        }
        self.transpose_x = True
        self.transpose_y = False


class TestMatmulCase13(TestMatmulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([4, 16]).astype("float32"),
            "y": np.random.random([16, 1]).astype("float32")
        }
        self.transpose_x = False
        self.transpose_y = False


class TestMatmulCase14(TestMatmulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([10, 1, 128, 64]).astype("float32"),
            "y": np.random.random([10, 12, 64, 128]).astype("float32")
        }
        self.transpose_x = False
        self.transpose_y = False


class TestMatmulCase15(TestMatmulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([10, 12, 128, 64]).astype("float32"),
            "y": np.random.random([10, 12, 128, 64]).astype("float32")
        }
        self.transpose_x = False
        self.transpose_y = True


class TestMatmulCase16(TestMatmulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([10, 12, 64, 128]).astype("float32"),
            "y": np.random.random([10, 12, 128, 64]).astype("float32")
        }
        self.transpose_x = True
        self.transpose_y = True


class TestMatmulCase17(TestMatmulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([128]).astype("float32"),
            "y": np.random.random([10, 12, 128, 64]).astype("float32")
        }
        self.transpose_x = False
        self.transpose_y = False


class TestMatmulCase18(TestMatmulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([128]).astype("float32"),
            "y": np.random.random([10, 12, 128, 64]).astype("float32")
        }
        self.transpose_x = True
        self.transpose_y = False


class TestMatmulCase19(TestMatmulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([128]).astype("float32"),
            "y": np.random.random([128]).astype("float32")
        }
        self.transpose_x = False
        self.transpose_y = False


class TestMatmulCase20(TestMatmulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([128]).astype("float32"),
            "y": np.random.random([128]).astype("float32")
        }
        self.transpose_x = True
        self.transpose_y = True


class TestMatmulTransposePass(TestMatmulOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([10, 1, 128, 64]).astype("float32"),
            "y": np.random.random([10, 12, 64, 128]).astype("float32")
        }
        self.transpose_x = False
        self.transpose_y = False
        self.perm = [0, 1, 3, 2]

    def paddle_func(self, x, y):
        out = paddle.matmul(
            x, y, transpose_x=self.transpose_x, transpose_y=self.transpose_y)
        return paddle.transpose(out, self.perm)

    def cinn_func(self, builder, x, y):
        out = builder.matmul(
            x, y, transpose_x=self.transpose_x, transpose_y=self.transpose_y)
        return builder.transpose(out, self.perm)


class TestMatmulTransposePassCase1(TestMatmulTransposePass):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([32, 64]).astype("float32"),
            "y": np.random.random([64, 128]).astype("float32")
        }
        self.transpose_x = False
        self.transpose_y = False
        self.perm = [1, 0]


if __name__ == "__main__":
    unittest.main()
