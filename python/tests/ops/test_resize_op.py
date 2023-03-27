#!/usr/bin/env python3

# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
from paddle.vision.transforms import functional as F


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestResizeOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x":
            np.array([[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10],
                        [11, 12, 13, 14, 15], [16, 17, 18, 19, 20],
                        [21, 22, 23, 24, 25]]]]).astype("int32")
        }
        self.outputs = {
            "y":
            np.array([[[[1, 2, 3, 4], [6, 7, 8, 9], [11, 12, 13, 14],
                        [16, 17, 18, 19]]]]).astype("int32")
        }
        self.out_shape = [4, 4]
        self.mode = "nearest"

    def build_paddle_program(self, target):
        y = paddle.to_tensor(self.outputs["y"], stop_gradient=False)
        self.paddle_outputs = [y]

    def build_cinn_program(self, target):
        builder = NetBuilder("resize")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape, "x")
        out = builder.resize(x, self.out_shape, self.mode)
        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x], [self.inputs["x"]], [out], passes=[])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestResizeCase1(TestResizeOp):
    def init_case(self):
        self.inputs = {
            "x":
            np.array([[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10],
                        [11, 12, 13, 14, 15], [16, 17, 18, 19, 20],
                        [21, 22, 23, 24, 25]]]]).astype("int32")
        }
        self.outputs = {
            "y":
            np.array([[[[1, 2, 3, 4], [7, 8, 9, 11], [13, 14, 16, 17],
                        [19, 21, 22, 23]]]]).astype("int32")
        }
        self.out_shape = [4, 4]
        self.mode = "bilinear"


class TestResizeCase2(TestResizeOp):
    def init_case(self):
        self.inputs = {
            "x":
            np.array([[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10],
                        [11, 12, 13, 14, 15], [16, 17, 18, 19, 20],
                        [21, 22, 23, 24, 25]]]]).astype("int32")
        }
        self.outputs = {
            "y":
            np.array([[[[1, 2, 3, 4], [7, 8, 9, 11], [13, 14, 16, 17],
                        [20, 21, 22, 23]]]]).astype("int32")
        }
        self.out_shape = [4, 4]
        self.mode = "bicubic"


class TestResizeCase3(TestResizeOp):
    def init_case(self):
        self.inputs = {
            "x":
            np.array([[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10],
                        [11, 12, 13, 14, 15], [16, 17, 18, 19, 20],
                        [21, 22, 23, 24, 25]]]]).astype("int32")
        }
        self.outputs = {
            "y":
            np.array([[[[1, 1, 2, 3, 4, 5], [1, 1, 2, 3, 4, 5],
                        [6, 6, 7, 8, 9, 10], [11, 11, 12, 13, 14, 15],
                        [16, 16, 17, 18, 19, 20], [21, 21, 22, 23, 24,
                                                   25]]]]).astype("int32")
        }
        self.out_shape = [6, 6]
        self.mode = "nearest"


class TestResizeCase4(TestResizeOp):
    def init_case(self):
        self.inputs = {
            "x":
            np.array([[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10],
                        [11, 12, 13, 14, 15], [16, 17, 18, 19, 20],
                        [21, 22, 23, 24, 25]]]]).astype("int32")
        }
        self.outputs = {
            "y":
            np.array([[[[1, 1, 2, 3, 4, 5], [5, 6, 6, 7, 8, 9],
                        [9, 10, 11, 11, 12, 13], [13, 14, 15, 16, 16, 17],
                        [17, 18, 19, 20, 20, 21], [21, 21, 22, 23, 24,
                                                   25]]]]).astype("int32")
        }
        self.out_shape = [6, 6]
        self.mode = "bilinear"


class TestResizeCase5(TestResizeOp):
    def init_case(self):
        self.inputs = {
            "x":
            np.array([[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10],
                        [11, 12, 13, 14, 15], [16, 17, 18, 19, 20],
                        [21, 22, 23, 24, 25]]]]).astype("int32")
        }
        self.outputs = {
            "y":
            np.array([[[[1, 1, 2, 3, 4, 5], [5, 5, 6, 7, 8, 9],
                        [9, 10, 11, 11, 12, 13], [13, 14, 15, 16, 16, 17],
                        [17, 18, 19, 20, 21, 21], [21, 22, 22, 23, 24,
                                                   25]]]]).astype("int32")
        }
        self.out_shape = [6, 6]
        self.mode = "bicubic"


if __name__ == "__main__":
    unittest.main()
