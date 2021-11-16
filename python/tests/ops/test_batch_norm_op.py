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
from op_test import OpTest, OpTestTool, random
import paddle
import paddle.nn.functional as F
import cinn
from cinn.frontend import *
from cinn.common import *


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestBatchNormOp(OpTest):
    def setUp(self):
        self.config()
        self.inputs = {
            "x": random(self.x_shape, self.dtype),
            "scale": random(self.param_shape, self.dtype),
            "bias": random(self.param_shape, self.dtype),
            "mean": random(self.param_shape, self.dtype),
            "variance": random(self.param_shape, self.dtype),
        }
        if self.backward:
            self.inputs["y_grad"] = random(self.x_shape, self.dtype)
        self.epsilon = 1e-05
        self.momentum = 0.9
        self.is_test = False

    def config(self):
        self.dtype = "float32"
        self.x_shape = [16, 32, 16, 16]
        self.param_shape = [self.x_shape[1]]
        self.data_format = "NCHW"
        self.backward = True

    def build_paddle_program(self, target):
        def _create_parameter(name):
            param = paddle.create_parameter(
                name=name,
                shape=self.param_shape,
                dtype=self.dtype,
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Assign(
                        self.inputs[name])))
            param.stop_gradient = True
            return param

        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        scale = paddle.to_tensor(self.inputs["scale"], stop_gradient=False)
        bias = paddle.to_tensor(self.inputs["bias"], stop_gradient=False)
        running_mean = _create_parameter("mean")
        running_variance = _create_parameter("variance")

        out = F.batch_norm(
            x=x,
            running_mean=running_mean,
            running_var=running_variance,
            weight=scale,
            bias=bias,
            epsilon=self.epsilon,
            momentum=self.momentum,
            training=not self.is_test,
            data_format=self.data_format)

        # Cannot get save_mean and save_variance of paddle.
        self.paddle_outputs = [out, None, None, running_mean, running_variance]
        if self.backward:
            self.paddle_grads = self.get_paddle_grads([out], [x, scale, bias],
                                                      [self.inputs["y_grad"]])

    def build_cinn_program(self, target):
        builder = NetBuilder("batch_norm")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        scale = builder.create_input(
            Float(32), self.inputs["scale"].shape, "scale")
        bias = builder.create_input(
            Float(32), self.inputs["bias"].shape, "bias")
        mean = builder.create_input(
            Float(32), self.inputs["mean"].shape, "mean")
        variance = builder.create_input(
            Float(32), self.inputs["variance"].shape, "variance")
        y, saved_mean, saved_variance, moving_mean, moving_variance = builder.batchnorm(
            x, scale, bias, mean, variance, self.epsilon, self.momentum,
            self.data_format, self.is_test)

        inputs = [x, scale, bias, mean, variance]
        feeds = [
            self.inputs["x"], self.inputs["scale"], self.inputs["bias"],
            self.inputs["mean"], self.inputs["variance"]
        ]
        outputs = [y, saved_mean, saved_variance, moving_mean, moving_variance]
        if self.backward:
            y_grad = builder.create_input(
                Float(32), self.inputs["y_grad"].shape, "y_grad")
            x_grad, scale_grad, bias_grad = builder.batch_norm_grad(
                y_grad, x, scale, saved_mean, saved_variance)

            inputs = inputs + [y_grad]
            feeds = feeds + [self.inputs["y_grad"]]
            outputs = outputs + [x_grad, scale_grad, bias_grad]

        prog = builder.build()
        outs = self.get_cinn_output(prog, target, inputs, feeds, outputs)

        self.cinn_outputs = [outs[0], outs[1], outs[2], outs[3], outs[4]]
        if self.backward:
            self.cinn_grads = [outs[5], outs[6], outs[7]]

    def test_check_results(self):
        self.check_outputs_and_grads(max_relative_error=1e-5)


if __name__ == "__main__":
    unittest.main()
