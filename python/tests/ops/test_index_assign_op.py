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
class TestIndexAssignOp(OpTest):
    def setUp(self):
        self.init_case()
        self.target = DefaultNVGPUTarget()

    def init_case(self):
        self.axis = 0
        self.inputs = {
            "x": np.random.random([10, 5]).astype("float32"),
            "y": np.random.random([3, 5]).astype("float32"),
            "index": np.random.randint(0, 10, size=3)
        }

    def build_paddle_program(self, target):
        x = self.inputs["x"]
        y = self.inputs["y"]

        out = x
        axis = self.axis
        while (axis < 0):
            axis += len(self.inputs["x"].shape)

        if axis == 0:
            for i in range(self.inputs["index"].shape[0]):
                out[self.inputs["index"][i]] = y[i]
        elif axis == 1:
            for i in range(self.inputs["x"].shape[0]):
                for j in range(self.inputs["index"].shape[0]):
                    out[i][self.inputs["index"][j]] = y[i][j]
        elif axis == 2:
            for i in range(self.inputs["x"].shape[0]):
                for j in range(self.inputs["x"].shape[1]):
                    for k in range(self.inputs["index"].shape[0]):
                        out[i][j][self.inputs["index"][k]] = y[i][j][k]
        else:
            self.assertTrue(False, "Axis {} No Implement".format(self.axis))

        pd_out = paddle.to_tensor(out, stop_gradient=True)
        self.paddle_outputs = [pd_out]

    def build_cinn_program(self, target):
        builder = NetBuilder("index_assign")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        y = builder.create_input(Float(32), self.inputs["y"].shape, "y")
        index = builder.create_input(
            Float(32), self.inputs["index"].shape, "index")
        out = builder.index_assign(x, y, index, self.axis)

        prog = builder.build()
        res = self.get_cinn_output(prog, set(), target, [x, y, index], [
            self.inputs["x"], self.inputs["y"],
            self.inputs["index"].astype("float32")
        ], [out])

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestIndexAssignCase1(TestIndexAssignOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([10, 5]).astype("float32"),
            "y": np.random.random([10, 3]).astype("float32"),
            "index": np.random.randint(0, 5, size=3)
        }
        self.axis = 1


class TestIndexAssignCase2(TestIndexAssignOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([10, 5, 5]).astype("float32"),
            "y": np.random.random([10, 5, 3]).astype("float32"),
            "index": np.random.randint(0, 5, size=3)
        }
        self.axis = -1


class TestIndexAssignCase3(TestIndexAssignOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([10]).astype("float32"),
            "y": np.random.random([1]).astype("float32"),
            "index": np.random.randint(0, 10, size=1)
        }
        self.axis = -1


class TestIndexAssignCase4(TestIndexAssignOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([10, 5]).astype("float32"),
            "y": np.random.random([3, 5]).astype("float32"),
            "index": np.array([0, 5, 0])
        }
        self.axis = 0


if __name__ == "__main__":
    unittest.main()
