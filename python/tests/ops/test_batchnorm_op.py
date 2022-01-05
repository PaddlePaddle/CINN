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
class TestBatchNorm(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x": np.random.random([1, 128, 16, 16]).astype("float32"),
            "running_mean": np.random.random([128]).astype("float32"),
            "running_var": np.random.random([128]).astype("float32"),
            "scale": np.random.random([128]).astype("float32"),
            "bias": np.random.random([128]).astype("float32"),
            "dy": np.random.random([1, 128, 16, 16]).astype("float32")
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        running_mean = paddle.to_tensor(
            self.inputs["running_mean"], stop_gradient=False)
        running_var = paddle.to_tensor(
            self.inputs["running_var"], stop_gradient=False)
        scale = paddle.to_tensor(self.inputs["scale"], stop_gradient=False)
        bias = paddle.to_tensor(self.inputs["bias"], stop_gradient=False)
        dy = paddle.to_tensor(self.inputs["dy"])

        # paddle.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, training=False, momentum=0.9, epsilon=1e-05, data_format='NCHW', name=None):
        y = paddle.nn.functional.batch_norm(
            x,
            running_mean,
            running_var,
            scale,
            bias,
            training=True,
            momentum=0.9,
            epsilon=1e-5,
            data_format="NCHW")
        paddle.autograd.backward([y], [dy], True)
        self.paddle_outputs = [
            y.numpy(),
            x.grad.numpy(),
            scale.grad.numpy(),
            bias.grad.numpy()
        ]

    def build_cinn_program(self, target):
        builder = NetBuilder("batchnnorm")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        running_mean = builder.create_input(
            Float(32), self.inputs["running_mean"].shape, "running_mean")
        running_var = builder.create_input(
            Float(32), self.inputs["running_var"].shape, "running_var")
        scale = builder.create_input(
            Float(32), self.inputs["scale"].shape, "scale")
        bias = builder.create_input(
            Float(32), self.inputs["bias"].shape, "bias")
        dy = builder.create_input(Float(32), self.inputs["dy"].shape, "dy")

        # outputs={y, saved_mean, saved_variance, moving_mean, moving_variance}
        y, mean, variance, running_mean_, running_var_ = builder.batchnorm(
            x, scale, bias, running_mean, running_var, is_test=False)
        # outputs=(x_grad, scale_grad, bias_grad)
        grad_x, grad_scale, grad_bias = builder.batch_norm_grad(
            dy, x, scale, mean, variance)
        prog = builder.build()

        res = self.get_cinn_output(
            prog, target, [x, scale, bias, running_mean, running_var, dy], [
                self.inputs["x"], self.inputs["scale"], self.inputs["bias"],
                self.inputs["running_mean"], self.inputs["running_var"],
                self.inputs["dy"]
            ], [y, grad_x, grad_scale, grad_bias])
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(1e-4)


if __name__ == "__main__":
    unittest.main()
