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


class TestClzOp(OpTest):

    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x": self.random([32, 64], 'int32')
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        self.paddle_outputs = [x]

    def build_cinn_program(self, target):
        builder = NetBuilder("clz")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape, "x"
        )
        out = builder.clz(x)
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])
        self.cinn_outputs = [res[0]]
        print(self.cinn_outputs)

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestClzCase1(TestClzOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([4, 3]).astype("int32")
        }


class TestClzCase2(TestClzOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([10, 100]).astype("uint32")
        }


class TestClzCase3(TestClzOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([4, 3, 5, 8]).astype("int64")
        }



class TestClzCase4(TestClzOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([10042]).astype("uint64")
        }


if __name__ == "__main__":
    unittest.main()
