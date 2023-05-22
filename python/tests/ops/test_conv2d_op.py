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

set_cinn_cudnn_deterministic(True)
paddle.set_flags({'FLAGS_cudnn_deterministic': 1})


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestConv2dNCHW(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x": self.random([3, 16, 32, 32], "float32"),
            "weight": self.random([16, 16, 3, 3], "float32"),
            "dy": self.random([3, 16, 30, 30], "float32")
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        weight = paddle.to_tensor(self.inputs["weight"], stop_gradient=False)

        y = paddle.nn.functional.conv2d(x, weight, data_format='NCHW')

        self.paddle_outputs = [y.numpy()]
        self.paddle_grads = self.get_paddle_grads([y], [x, weight],
                                                  [self.inputs["dy"]])

    def build_cinn_program(self, target):
        builder = NetBuilder("conv2d")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape, "x")
        weight = builder.create_input(
            self.nptype2cinntype(self.inputs["weight"].dtype),
            self.inputs["weight"].shape, "weight")
        dy = builder.create_input(
            self.nptype2cinntype(self.inputs["dy"].dtype),
            self.inputs["dy"].shape, "dy")

        y = builder.conv2d(x, weight, data_format='NCHW')

        x_grad = builder.conv(
            weight,
            dy,
            data_format='NCHW',
            conv_type="backward_data",
            output_shape=x.shape())
        weight_grad = builder.conv(
            x,
            dy,
            data_format='NCHW',
            conv_type="backward_filter",
            output_shape=weight.shape())

        prog = builder.build()

        res = self.get_cinn_output(
            prog,
            target, [x, weight, dy],
            [self.inputs["x"], self.inputs["weight"], self.inputs["dy"]],
            [y, x_grad, weight_grad],
            passes=[])

        self.cinn_outputs = [res[0]]
        self.cinn_grads = [res[1], res[2]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestConv2dNCHWFP16(TestConv2dNCHW):
    def init_case(self):
        self.inputs = {
            "x": self.random([3, 16, 32, 32], "float16"),
            "weight": self.random([16, 16, 3, 3], "float16"),
            "dy": self.random([3, 16, 30, 30], "float16")
        }

    def test_check_results(self):
        self.check_outputs_and_grads(max_relative_error=1e-3)


@OpTestTool.skip_if(not is_compiled_with_cudnn(),
                    "conv2d NHWC only support cudnn now.")
class TestConv2dNHWC(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x": self.random([3, 32, 32, 16], "float32"),
            "weight": self.random([16, 16, 3, 3], "float32"),
            "dy": self.random([3, 30, 30, 16], "float32")
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        weight = paddle.to_tensor(self.inputs["weight"], stop_gradient=False)

        y = paddle.nn.functional.conv2d(x, weight, data_format='NHWC')

        self.paddle_outputs = [y.numpy()]
        self.paddle_grads = self.get_paddle_grads([y], [x, weight],
                                                  [self.inputs["dy"]])

    def build_cinn_program(self, target):
        builder = NetBuilder("conv2d")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape, "x")
        weight = builder.create_input(
            self.nptype2cinntype(self.inputs["weight"].dtype),
            self.inputs["weight"].shape, "weight")
        dy = builder.create_input(
            self.nptype2cinntype(self.inputs["dy"].dtype),
            self.inputs["dy"].shape, "dy")

        w_t = builder.transpose(weight, [0, 2, 3, 1])

        y = builder.conv2d(x, w_t, data_format='NHWC')

        x_grad = builder.conv(
            w_t,
            dy,
            data_format='NHWC',
            conv_type="backward_data",
            output_shape=x.shape())
        w_grad = builder.conv(
            x,
            dy,
            data_format='NHWC',
            conv_type="backward_filter",
            output_shape=w_t.shape())

        weight_grad = builder.transpose(w_grad, [0, 3, 1, 2])

        prog = builder.build()

        res = self.get_cinn_output(
            prog,
            target, [x, weight, dy],
            [self.inputs["x"], self.inputs["weight"], self.inputs["dy"]],
            [y, x_grad, weight_grad],
            passes=[])

        self.cinn_outputs = [res[0]]
        self.cinn_grads = [res[1], res[2]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestConv2dNHWCFP16(TestConv2dNHWC):
    def init_case(self):
        self.inputs = {
            "x": self.random([3, 32, 32, 16], "float16"),
            "weight": self.random([16, 16, 3, 3], "float16"),
            "dy": self.random([3, 30, 30, 16], "float16")
        }

    def test_check_results(self):
        self.check_outputs_and_grads(max_relative_error=1e-3)


if __name__ == "__main__":
    unittest.main()
