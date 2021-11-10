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
import cinn
import numpy as np
import paddle
import unittest

from cinn.frontend import *
from cinn.common import *
from cinn.runtime import *
from op_test import OpTest, OpTestTool


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestBatchNormOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x": np.random.random([4, 64, 32, 32]).astype("float32"),
            "scale": np.random.random([64]).astype("float32"),
            "bias": np.random.random([64]).astype("float32"),
            "running_mean": np.random.random([64]).astype("float32"),
            "running_variance": np.random.random([64]).astype("float32"),
            "dy": np.random.random([4, 64, 32, 32]).astype("float32")
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        scale = paddle.to_tensor(self.inputs["scale"], stop_gradient=False)
        bias = paddle.to_tensor(self.inputs["bias"], stop_gradient=False)
        mean = paddle.to_tensor(self.inputs["running_mean"])
        variance = paddle.to_tensor(self.inputs["running_variance"])
        dy = paddle.to_tensor(self.inputs["dy"], stop_gradient=False)

        y = paddle.nn.functional.batch_norm(
            x,
            mean,
            variance,
            scale,
            bias,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW')

        paddle.autograd.backward([y], [dy], True)
        self.paddle_outputs = [
            y.numpy(),
            x.grad.numpy(),
            scale.grad.numpy(),
            bias.grad.numpy()
        ]

    def build_cinn_program(self, target):
        builder = NetBuilder("batchnorm")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        scale = builder.create_input(
            Float(32), self.inputs["scale"].shape, "scale")
        bias = builder.create_input(
            Float(32), self.inputs["bias"].shape, "bias")
        mean = builder.create_input(
            Float(32), self.inputs["running_mean"].shape, "running_mean")
        variance = builder.create_input(
            Float(32), self.inputs["running_variance"].shape,
            "running_variance")
        dy = builder.create_input(Float(32), self.inputs["dy"].shape, "dy")

        y = builder.batchnorm(x, scale, bias, mean, variance, 1e-5, 0.9,
                              "NCHW", False)
        grad = builder.batch_norm_grad(dy, x, scale, y[1], y[2])
        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, scale, bias, mean, variance, dy], [
                self.inputs["x"], self.inputs["scale"], self.inputs["bias"],
                self.inputs["running_mean"], self.inputs["running_variance"],
                self.inputs["dy"]
            ], [y[0], y[1], y[2], y[3], y[4], grad[0], grad[1], grad[2]])

        self.cinn_outputs = [res[0], res[5], res[6], res[7]]

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
