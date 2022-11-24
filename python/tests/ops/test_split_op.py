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
class TestSplitOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {"x": np.random.random([9, 9, 5]).astype("float32")}
        self.num_or_sections = [2, 3, 4]
        self.axis = 0

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)

        if (len(self.num_or_sections) == 1):
            num = self.num_or_sections[0]
        else:
            num = self.num_or_sections

        out = paddle.split(x, num_or_sections=num, axis=self.axis)

        self.paddle_outputs = out

    def build_cinn_program(self, target):
        builder = NetBuilder("split")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        out = builder.split(
            x, num_or_sections=self.num_or_sections, axis=self.axis)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], out)

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestSplitCase1(TestSplitOp):
    def init_case(self):
        self.inputs = {"x": np.random.random([9, 9, 5]).astype("float32")}
        self.num_or_sections = [3]
        self.axis = 0


class TestSplitCase2(TestSplitOp):
    def init_case(self):
        self.inputs = {"x": np.random.random([9, 9, 5]).astype("float32")}
        self.num_or_sections = [3]
        self.axis = 1


class TestSplitCase3(TestSplitOp):
    def init_case(self):
        self.inputs = {"x": np.random.random([9, 9, 5]).astype("float32")}
        self.num_or_sections = [2, 3, -1]
        self.axis = 1


class TestSplitCase4(TestSplitOp):
    def init_case(self):
        self.inputs = {"x": np.random.random([8, 9, 5]).astype("float32")}
        self.num_or_sections = [2]
        self.axis = 0


class TestSplitCase5(TestSplitOp):
    def init_case(self):
        self.inputs = {"x": np.random.random([8, 9, 5]).astype("float32")}
        self.num_or_sections = [-1, 2, 2, 2]
        self.axis = 0


class TestSplitCase6(TestSplitOp):
    def init_case(self):
        self.inputs = {"x": np.random.random([2048, 9, 6]).astype("float32")}
        self.num_or_sections = [2]
        self.axis = 2


class TestSplitCase7(TestSplitOp):
    def init_case(self):
        self.inputs = {
            "x": np.random.random([10, 128, 4096]).astype("float32")
        }
        self.num_or_sections = [2]
        self.axis = 2


if __name__ == "__main__":
    unittest.main()
