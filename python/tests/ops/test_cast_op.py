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

import cinn
import numpy as np
import paddle
import unittest

from cinn.frontend import *
from cinn.common import *
from op_test import OpTest, OpTestTool


class TestCastOp(OpTest):
    def setUp(self):
        self.init_case()
        self.inputs = {"x": np.random.randint(0, 10, [2, 3])}

    def init_case(self):
        self.bit_width = 32

    def build_paddle_program(self, target):
        x = paddle.to_tensor(
            self.inputs["x"].astype("float" + str(self.bit_width)),
            stop_gradient=True)
        out1 = paddle.cast(x, "int" + str(self.bit_width))
        x = paddle.to_tensor(
            self.inputs["x"].astype("int" + str(self.bit_width)),
            stop_gradient=True)
        out2 = paddle.cast(x, "float" + str(self.bit_width))
        self.paddle_outputs = [out1, out2]
        print(self.paddle_outputs)

    def build_cinn_program(self, target):
        builder = NetBuilder("cast_test")
        x = builder.create_input(
            Float(self.bit_width), self.inputs["x"].shape, "x")
        out = builder.cast(x, "int" + str(self.bit_width))

        prog = builder.build()
        res1 = self.get_cinn_output(
            prog, target, [x],
            [self.inputs["x"].astype("float" + str(self.bit_width))], [out])

        builder = NetBuilder("cast_test")
        x = builder.create_input(
            Int(self.bit_width), self.inputs["x"].shape, "x")
        out = builder.cast(x, "float" + str(self.bit_width))

        prog = builder.build()
        res2 = self.get_cinn_output(
            prog, target, [x],
            [self.inputs["x"].astype("int" + str(self.bit_width))], [out])
        self.cinn_outputs = [res1[0], res2[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
