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
from op_test import OpTest
import paddle
import paddle.nn.functional as F

from cinn.frontend import *
from cinn import Target
from cinn.framework import *
import cinn
from cinn import runtime
from cinn import ir
from cinn import lang
from cinn.common import *


class TestReluOp(OpTest):
    def setUp(self):
        self.init_case()
        self.init_target()

    def init_case(self):
        self.inputs = {"x": np.random.random([4, 4]).astype("float32")}

    def build_paddle_program(self):
        x = paddle.to_tensor(self.inputs["x"])
        out = F.relu(x)
        grads = self.get_paddle_grads([out], [x])

        return tuple(out), grads

    def build_cinn_program(self):
        builder = NetBuilder("relu")
        x = builder.create_input(Float(32), (4, 4), "x")
        out = builder.relu(x)
        x_grad = builder.relu_grad()
        prog = builder.build()
        result = self.get_cinn_output(prog, [x], [self.inputs["x"]], out)

        return tuple(result)

    def test_check_output(self):
        self.check_output()

    def test_check_grad(slef):
        self.check_grad()


if __name__ == "__main__":
    unittest.main()
