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

import sys
import unittest
import numpy as np
from op_test import OpTest, OpTestTool
import paddle
from cinn.frontend import *
from cinn.common import *


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestGatherNdOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            'x': self.random([2, 3, 4], 'float32'),
            'index': np.array([[1]], dtype='int32')
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        index = paddle.to_tensor(self.inputs["index"], stop_gradient=False)
        out = paddle.gather_nd(x, index)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("GatherNd")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape, "x")
        index = builder.create_input(
            self.nptype2cinntype(self.inputs["index"].dtype),
            self.inputs["index"].shape, "index")
        out = builder.gather_nd(x, index, [])

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x, index],
                                   [self.inputs["x"], self.inputs["index"]],
                                   [out])

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestGatherNdCase1(TestGatherNdOp):
    def init_case(self):
        self.inputs = {
            'x': self.random([2, 3, 4], 'float32'),
            'index': np.array([[0, 2]], dtype='int32')
        }


class TestGatherNdCase2(TestGatherNdOp):
    def init_case(self):
        self.inputs = {
            'x': self.random([2, 3, 4], 'float32'),
            'index': np.array([[1, 2, 3]], dtype='int32')
        }


if __name__ == "__main__":
    unittest.main()
