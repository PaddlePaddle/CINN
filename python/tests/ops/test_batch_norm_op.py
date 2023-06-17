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
from op_test_helper import TestCaseHelper
import paddle
import cinn
from cinn.frontend import *
from cinn.common import *


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestBatchNormTrainOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.case["x_shape"], dtype=self.case["x_dtype"])

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np)
        batch_norm = paddle.nn.BatchNorm(
            self.case["x_shape"][1], act=None, is_test=False)
        out = batch_norm(x)

        self.paddle_outputs = [out]

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("batch_norm")
        x = builder.create_input(
            self.nptype2cinntype(self.case["x_dtype"]), self.case["x_shape"],
            "x")
        scale = builder.fill_constant([self.case["x_shape"][1]], 1.0, 'scale',
                                      'float32')
        bias = builder.fill_constant([self.case["x_shape"][1]], 0.0, 'bias',
                                     'float32')
        mean = builder.fill_constant([self.case["x_shape"][1]], 0.0, 'mean',
                                     'float32')
        variance = builder.fill_constant([self.case["x_shape"][1]], 1.0,
                                         'variance', 'float32')

        out = builder.batchnorm(x, scale, bias, mean, variance, is_test=False)

        prog = builder.build()
        forward_res = self.get_cinn_output(
            prog, target, [x], [self.x_np], out, passes=[])
        self.cinn_outputs = [forward_res[0]]

    def test_check_results(self):
        max_relative_error = self.case[
            "max_relative_error"] if "max_relative_error" in self.case else 1e-5
        self.check_outputs_and_grads(max_relative_error=max_relative_error)


class TestBatchNormTrainOpAll(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestBatchNormTrainOpCase"
        self.cls = TestBatchNormTrainOp

        self.inputs = [
            {
                "x_shape": [2, 16, 8, 8],
            },
            {
                "x_shape": [2, 16, 8, 1],
            },
            {
                "x_shape": [2, 16, 2048, 8],
            },
        ]
        self.dtypes = [
            {
                "x_dtype": "float16",
                "max_relative_error": 1e-5
            },
            {
                "x_dtype": "float32",
                "max_relative_error": 1e-3
            },
            {
                "x_dtype": "bfloat16",
                "max_relative_error": 1e-2
            },
        ]
        self.attrs = []


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestBatchNormBackwardOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = [{
            "x":
            self.random([2, 16, 8, 8], "float32", 0.0, 10.0),
            "dout":
            self.random([2, 16, 8, 8], "float32", 1e-7, 1e-6),
            "num_channels":
            16
        }]

    def build_paddle_program(self, target):
        for inputs in self.inputs:
            x = paddle.to_tensor(inputs["x"], stop_gradient=False)
            batch_norm = paddle.nn.BatchNorm(
                inputs["num_channels"], act=None, is_test=False)
            out = batch_norm(x)

            self.paddle_outputs.append(out)
            grad = self.get_paddle_grads([out], [x], [inputs["dout"]])
            self.paddle_grads.append(grad[0])

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        for inputs in self.inputs:
            builder = NetBuilder("batch_norm")
            x = builder.create_input(
                self.nptype2cinntype(inputs["x"].dtype), inputs["x"].shape,
                "x")
            scale = builder.fill_constant([inputs["num_channels"]], 1.0,
                                          'scale', 'float32')
            bias = builder.fill_constant([inputs["num_channels"]], 0.0, 'bias',
                                         'float32')
            mean = builder.fill_constant([inputs["num_channels"]], 0.0, 'mean',
                                         'float32')
            variance = builder.fill_constant([inputs["num_channels"]], 1.0,
                                             'variance', 'float32')

            out = builder.batchnorm(
                x, scale, bias, mean, variance, is_test=False)

            prog = builder.build()
            forward_res = self.get_cinn_output(
                prog, target, [x], [inputs["x"]], out, passes=[])
            self.cinn_outputs.append(forward_res[0])

            builder_grad = NetBuilder("batch_norm_grad")
            dout = builder_grad.create_input(
                self.nptype2cinntype(inputs["dout"].dtype),
                inputs["dout"].shape, "dout")
            x_g = builder_grad.create_input(
                self.nptype2cinntype(inputs["x"].dtype), inputs["x"].shape,
                "x_g")
            scale_g = builder_grad.fill_constant(scale.shape(), 1.0, 'scale_g',
                                                 'float32')
            save_mean = builder_grad.create_input(
                self.nptype2cinntype('float32'), out[1].shape(), "save_mean")
            save_variance = builder_grad.create_input(
                self.nptype2cinntype('float32'), out[2].shape(),
                "save_variance")

            out_grad = builder_grad.batch_norm_grad(dout, x_g, scale_g,
                                                    save_mean, save_variance)
            prog = builder_grad.build()
            backward_res = self.get_cinn_output(
                prog,
                target, [dout, x_g, save_mean, save_variance],
                [inputs["dout"], inputs["x"], forward_res[1], forward_res[2]],
                out_grad,
                passes=[])
            self.cinn_grads.append(backward_res[0])

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestBatchNormBackwardAll(TestBatchNormBackwardOp):
    def init_case(self):
        self.inputs = []
        for x_shape in [[2, 16, 8, 8], [2, 16, 8, 1], [2, 16, 2048, 8]]:
            for x_type in ["float16", "float32"]:
                self.inputs.append({
                    "x":
                    self.random(x_shape, x_type, 0.0, 1.0),
                    "dout":
                    self.random(x_shape, x_type, 1e-7, 1e-6),
                    "num_channels":
                    x_shape[1]
                })

    def test_check_results(self):
        self.check_outputs_and_grads(max_relative_error=1e-3)


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestBatchNormInferOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.num_channels = 16
        self.inputs = {
            "x": self.random([2, self.num_channels, 8, 8], "float32", 0.0,
                             1.0),
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"])
        batch_norm = paddle.nn.BatchNorm(
            self.num_channels, act=None, is_test=True)
        out = batch_norm(x)

        self.paddle_outputs = [out]

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
        variance = builder.fill_constant([self.num_channels], 1.0, 'variance',
                                         'float32')

        out = builder.batchnorm(x, scale, bias, mean, variance, is_test=False)

        prog = builder.build()
        forward_res = self.get_cinn_output(
            prog, target, [x], [self.inputs["x"]], out, passes=[])
        self.cinn_outputs = [forward_res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    TestBatchNormTrainOpAll().run()
