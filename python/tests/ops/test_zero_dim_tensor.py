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


# 1) x is 0D, y is 0D
@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestElementwiseAddOp(OpTest):
    def setUp(self):
        self.init_input()

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype("float32"),
            "y": np.random.randint(-10, 10, []).astype("float32"),
            "dout": np.random.randint(-10, 10, []).astype("float32"),
        }

    def assertShapeEqual(self, res):
        out, x_grad, y_grad = res
        self.assertEqual(out.shape, ())
        self.assertEqual(x_grad.shape, ())
        self.assertEqual(y_grad.shape, ())

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=False)
        out = paddle.add(x, y)

        self.paddle_outputs = [out]
        self.paddle_grads = self.get_paddle_grads([out], [x, y],
                                                  [self.inputs["dout"]])

    def build_cinn_program(self, target):
        builder = NetBuilder("add")
        import sys
        sys.stdout.flush()
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        y = builder.create_input(Float(32), self.inputs["y"].shape, "y")
        out = builder.add(x, y)

        dout = builder.create_input(
            Float(32), self.inputs["dout"].shape, "dout")
        x_grad, y_grad = builder.elementwise_add_grad(dout, x, y)

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, y, dout],
            [self.inputs["x"], self.inputs["y"], self.inputs["dout"]],
            [out, x_grad, y_grad])
        self.assertShapeEqual(res)

        self.cinn_outputs = [res[0]]
        self.cinn_grads = [res[1], res[2]]

    def test_check_results(self):
        self.check_outputs_and_grads()


# 2) x is ND, y is 0D
# NOTE: CINN only supports x's rank >= y's rank, hence no need to test next scenario: `3) x is 0D, y is ND`
@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestElementwiseAddOp2(TestElementwiseAddOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, [3, 5]).astype("float32"),
            "y": np.random.randint(-10, 10, []).astype("float32"),
            "dout": np.random.randint(-10, 10, [3, 5]).astype("float32"),
        }

    def assertShapeEqual(self, res):
        out, x_grad, y_grad = res
        self.assertEqual(out.shape, (3, 5))
        self.assertEqual(x_grad.shape, (3, 5))
        self.assertEqual(y_grad.shape, ())


if __name__ == "__main__":
    unittest.main()
