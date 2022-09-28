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
import paddle
import cinn
from cinn.frontend import *
from cinn.common import *
import logging
import os


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestSelectOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "Condition": np.zeros(100).astype('bool'),
            "X": np.random.uniform((-3), 5, 100).astype('float32'),
            "Y": np.random.uniform((-3), 5, 100).astype('float32')
        }

    def build_paddle_program(self, target):
        c = paddle.to_tensor(self.inputs["Condition"], stop_gradient=True)
        x = paddle.to_tensor(self.inputs["X"], stop_gradient=True)
        y = paddle.to_tensor(self.inputs["Y"], stop_gradient=True)

        out = paddle.where(c, x, y)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("select")
        c = builder.create_input(Bool(), self.inputs["Condition"].shape,
                                 "Condition")
        x = builder.create_input(Float(32), self.inputs["X"].shape, "X")
        y = builder.create_input(Float(32), self.inputs["Y"].shape, "Y")

        out = builder.select(c, x, y)
        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [c, x, y],
            [self.inputs["Condition"], self.inputs["X"], self.inputs["Y"]],
            [out])
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestSelectOp1(TestSelectOp):
    def init_config(self):
        self.x = np.random.uniform((-5), 5, (60, 2)).astype('float32')
        self.y = np.random.uniform((-5), 5, (60, 2)).astype('float32')
        self.cond = np.ones((60, 2)).astype('bool')


class TestSelectOp2(TestSelectOp):
    def init_config(self):
        self.x = np.random.uniform((-3), 5, (20, 2, 4)).astype('float32')
        self.y = np.random.uniform((-3), 5, (20, 2, 4)).astype('float32')
        self.cond = np.array(np.random.randint(2, size=(20, 2, 4)), dtype=bool)


if __name__ == "__main__":
    unittest.main()
