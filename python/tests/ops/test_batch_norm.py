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

import unittest, sys
import numpy as np
from op_test import OpTest, OpTestTool
import paddle
import cinn
from cinn.frontend import *
from cinn.common import *


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestBatchNormOp(OpTest):
    def setUp(self):
        paddle.seed(1234)
        np.random.seed(1234)
        self.init_case()

    def init_case(self):
        self.num_channels = 256
        self.inputs = {
            "x": self.random([2, 256, 55, 55], "float32"),
            "dout": self.random([2, 256, 55, 55], "float32"),
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        batch_norm = paddle.nn.BatchNorm(
            self.num_channels, act=None, is_test=False)
        out = batch_norm(x)

        self.paddle_outputs = [out]
        self.paddle_grads = self.get_paddle_grads([out], [x],
                                                  [self.inputs["dout"]])

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("batch_norm")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape, "x")
        scale = builder.fill_constant([self.num_channels], 1.0, 'scale',
                                      'float32')
        bias = builder.fill_constant([self.num_channels], 0.0, 'bias',
                                     'float32')
        mean = builder.fill_constant([self.num_channels], 0.0, 'mean',
                                     'float32')
        variance = builder.fill_constant([self.num_channels], 0.0, 'variance',
                                         'float32')

        out = builder.batchnorm(x, scale, bias, mean, variance, is_test=False)

        prog = builder.build()
        forward_res = self.get_cinn_output(
            prog, target, [x], [self.inputs["x"]], out, passes=[])
        self.cinn_outputs = [forward_res[0]]

        builder_grad = NetBuilder("batch_norm_grad")
        dout = builder_grad.create_input(
            self.nptype2cinntype(self.inputs["dout"].dtype),
            self.inputs["dout"].shape, "dout")
        x_g = builder_grad.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape, "x_g")
        scale_g = builder_grad.fill_constant(scale.shape(), 1.0, 'scale_g',
                                             'float32')
        save_mean = builder_grad.create_input(
            self.nptype2cinntype('float32'), out[1].shape(), "save_mean")
        save_variance = builder_grad.create_input(
            self.nptype2cinntype('float32'), out[2].shape(), "save_variance")

        out_grad = builder_grad.batch_norm_grad(dout, x_g, scale_g, save_mean,
                                                save_variance)
        prog = builder_grad.build()
        backward_res = self.get_cinn_output(
            prog,
            target, [dout, x_g, save_mean, save_variance], [
                self.inputs["dout"], self.inputs["x"], forward_res[1],
                forward_res[2]
            ],
            out_grad,
            passes=[])
        self.cinn_grads = [backward_res[0]]

    def test_check_results(self):
        # TODO(thisjiang): remove max_relative_error after batch_norm_grad accuracy bug fixed
        self.check_outputs_and_grads(max_relative_error=1.0)


# Reopen after decomposer infer dtype fixed
class TestBatchNormFP16(TestBatchNormOp):
    def init_case(self):
        self.num_channels = 256
        self.inputs = {
            "x": self.random([2, 256, 55, 55], "float16"),
            "dout": self.random([2, 256, 55, 55], "float16"),
        }

    def test_check_results(self):
        self.check_outputs_and_grads(max_relative_error=1e-2)


if __name__ == "__main__":
    unittest.main()
