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


##############################
## Test ElementwiseAddGrad  ##
##############################
# 1) x is 0D, y is 0D
@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestElementwiseAddGrad(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.init_input()

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype("float32"),
            "y": np.random.randint(-10, 10, []).astype("float32"),
            "dout": np.random.randint(-10, 10, []).astype("float32"),
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=False)
        out = paddle.add(x, y)

        self.paddle_outputs = [out]
        self.paddle_grads = self.get_paddle_grads([out], [x, y],
                                                  [self.inputs["dout"]])

    def build_cinn_program(self, target):
        builder = NetBuilder("add")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        y = builder.create_input(Float(32), self.inputs["y"].shape, "y")
        # Test elementwise_add here, next unittest tests add, actually these two APIs are same.
        out = builder.elementwise_add(x, y)

        dout = builder.create_input(
            Float(32), self.inputs["dout"].shape, "dout")
        x_grad, y_grad = builder.elementwise_add_grad(dout, x, y)

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, y, dout],
            [self.inputs["x"], self.inputs["y"], self.inputs["dout"]],
            [out, x_grad, y_grad])

        out, x_grad, y_grad = res
        self.cinn_outputs = [out]
        self.cinn_grads = [x_grad, y_grad]
        self.assertEqual(out.shape, self.inputs["dout"].shape)
        self.assertEqual(x_grad.shape, self.inputs["x"].shape)
        self.assertEqual(y_grad.shape, self.inputs["y"].shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


# 2) x is ND, y is 0D
# NOTE: CINN only supports x's rank >= y's rank, hence no need to test next scenario: `3) x is 0D, y is ND`
class TestElementwiseAddGrad1(TestElementwiseAddGrad):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, [3, 5]).astype("float32"),
            "y": np.random.randint(-10, 10, []).astype("float32"),
            "dout": np.random.randint(-10, 10, [3, 5]).astype("float32"),
        }


#############################
#### Test ElementwiseOp  ####
#############################
# 1) x is 0D, y is 0D
@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestElementwiseOp(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.init_input()

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype("float32"),
            "y": np.random.randint(-10, 10, []).astype("float32"),
        }
        self.target_shape = ()

    def paddle_func(self, x, y):
        return paddle.add(x, y)

    def cinn_func(self, builder, x, y):
        return builder.add(x, y)

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=False)
        out = self.paddle_func(x, y)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("elementwise_op")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        y = builder.create_input(Float(32), self.inputs["y"].shape, "y")
        out = self.cinn_func(builder, x, y)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x, y],
                                   [self.inputs["x"], self.inputs["y"]], [out])

        self.cinn_outputs = res
        self.assertEqual(res[0].shape, self.target_shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestElementwiseAdd(TestElementwiseOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, [3, 5]).astype("float32"),
            "y": np.random.randint(-10, 10, []).astype("float32"),
        }
        self.target_shape = (3, 5)


class TestElementwiseSub1(TestElementwiseOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype("float32"),
            "y": np.random.randint(-10, 10, []).astype("float32"),
        }
        self.target_shape = ()

    def paddle_func(self, x, y):
        return paddle.subtract(x, y)

    def cinn_func(self, builder, x, y):
        return builder.subtract(x, y)


class TestElementwiseSub2(TestElementwiseSub1):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, [3, 5]).astype("float32"),
            "y": np.random.randint(-10, 10, []).astype("float32"),
        }
        self.target_shape = (3, 5)


class TestElementwiseMul1(TestElementwiseOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype("float32"),
            "y": np.random.randint(-10, 10, []).astype("float32"),
        }
        self.target_shape = ()

    def paddle_func(self, x, y):
        return paddle.multiply(x, y)

    def cinn_func(self, builder, x, y):
        return builder.multiply(x, y)


class TestElementwiseMul2(TestElementwiseMul1):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, [3, 5]).astype("float32"),
            "y": np.random.randint(-10, 10, []).astype("float32"),
        }
        self.target_shape = (3, 5)


class TestElementwiseMul3(TestElementwiseOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype("float32"),
            "y": np.random.randint(-10, 10, []).astype("float32"),
        }
        self.target_shape = ()

    def paddle_func(self, x, y):
        return paddle.multiply(x, y)

    def cinn_func(self, builder, x, y):
        return builder.elementwise_mul(x, y)


class TestElementwiseMul4(TestElementwiseMul3):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, [3, 5]).astype("float32"),
            "y": np.random.randint(-10, 10, []).astype("float32"),
        }
        self.target_shape = (3, 5)


class TestElementwiseDiv1(TestElementwiseOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype("float32"),
            "y": np.random.randint(-10, 10, []).astype("float32"),
        }
        self.target_shape = ()

    def paddle_func(self, x, y):
        return paddle.divide(x, y)

    def cinn_func(self, builder, x, y):
        return builder.divide(x, y)


class TestElementwiseDiv2(TestElementwiseDiv1):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, [3, 5]).astype("float32"),
            "y": np.random.randint(-10, 10, []).astype("float32"),
        }
        self.target_shape = (3, 5)


if __name__ == "__main__":
    unittest.main()
