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
import paddle.nn.functional as F
import cinn
from cinn.frontend import *
from cinn.common import *


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestBatchNormOp(OpTest):
    def setUp(self):
        def _random(shape, dtype):
            return np.random.random(shape).astype(dtype)

        self.config()
        self.inputs = {
            "x": _random(self.x_shape, self.dtype),
            "scale": _random(self.param_shape, self.dtype),
            "bias": _random(self.param_shape, self.dtype),
            "moving_mean": _random(self.param_shape, self.dtype),
            "moving_variance": _random(self.param_shape, self.dtype),
        }

    def config(self):
        self.dtype = "float32"
        self.x_shape = [4, 16, 4, 4]
        self.param_shape = [16]
        self.epsilon = 1e-05
        self.momentum = 0.9
        self.data_format = "NCHW"

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
        running_mean = _create_parameter("moving_mean")
        running_variance = _create_parameter("moving_variance")

        out = F.batch_norm(
            x=x,
            running_mean=running_mean,
            running_var=running_variance,
            weight=scale,
            bias=bias,
            epsilon=self.epsilon,
            momentum=self.momentum,
            training=True,
            data_format=self.data_format)

        # Cannot get save_mean and save_variance of paddle.
        self.paddle_outputs = [out, None, None, running_mean, running_variance]

    def build_cinn_program(self, target):
        builder = NetBuilder("batch_norm")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        scale = builder.create_input(
            Float(32), self.inputs["scale"].shape, "scale")
        bias = builder.create_input(
            Float(32), self.inputs["bias"].shape, "bias")
        mean = builder.create_input(
            Float(32), self.inputs["moving_mean"].shape, "moving_mean")
        variance = builder.create_input(
            Float(32), self.inputs["moving_variance"].shape, "moving_variance")
        outs = builder.batch_norm_train(x, scale, bias, mean, variance)
        prog = builder.build()
        forward_res = self.get_cinn_output(
            prog, target, [x, scale, bias, mean, variance], [
                self.inputs["x"], self.inputs["scale"], self.inputs["bias"],
                self.inputs["moving_mean"], self.inputs["moving_variance"]
            ], outs)

        self.cinn_outputs = forward_res

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
