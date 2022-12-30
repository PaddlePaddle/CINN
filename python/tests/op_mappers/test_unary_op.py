#!/usr/bin/env python3

# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

import sys
import unittest
import numpy as np
from op_mapper_test import OpMapperTest
import paddle
from cinn.frontend import *
from cinn.common import *

paddle.enable_static()

enable_gpu = sys.argv.pop()


class TestUnaryOp(OpMapperTest):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
            self.place = paddle.CUDAPlace(0)
        else:
            self.target = DefaultHostTarget()
            self.place = paddle.CPUPlace()

    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], "float32")}

    def set_unary_func(self, x):
        return paddle.sqrt(x)

    def set_paddle_program(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype)
        out = self.set_unary_func(x)

        return ([x.name], [out])

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestSqrtOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.sqrt(x)


class TestGeluOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.gelu(x)


class TestSigmoidOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.sigmoid(x)


class TestExpOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.exp(x)


class TestErfOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.erf(x)


class TestRsqrtOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.rsqrt(x)


class TestSinOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.sin(x)


class TestCosOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.cos(x)


class TestTanOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.tan(x)


class TestSinhOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.sinh(x)


class TestCoshOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.cosh(x)


class TestTanhOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.tanh(x)


class TestAsinOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.asin(x)


class TestAcosOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.acos(x)


class TestAtanOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.atan(x)


class TestAsinhOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.asinh(x)


class TestAcoshOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.acosh(x)


class TestAtanhOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.atanh(x)


class TestSignOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.sign(x)


class TestAbsOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.abs(x)


class TestReciprocalOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.reciprocal(x)


class TestFloorOp(TestUnaryOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], "float32", -2.0, 2.0)}

    def set_unary_func(self, x):
        return paddle.floor(x)

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestCeilOp(TestUnaryOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], "float32", -2.0, 2.0)}

    def set_unary_func(self, x):
        return paddle.ceil(x)

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestRoundOp(TestUnaryOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], "float32", -2.0, 2.0)}

    def set_unary_func(self, x):
        return paddle.round(x)

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestTruncOp(TestUnaryOp):
    def init_input_data(self):
        self.feed_data = {'x': self.random([32, 64], "float32", -2.0, 2.0)}

    def set_unary_func(self, x):
        return paddle.trunc(x)

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestIsNanOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.isnan(x)

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestIsNanCase1(TestIsNanOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64])}
        self.inputs["x"][0] = [np.nan] * 64


class TestIsFiniteOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.isfinite(x)

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestIsFiniteCase1(TestIsFiniteOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64])}
        self.inputs["x"][0] = [np.inf] * 64


class TestIsInfOp(TestUnaryOp):
    def set_unary_func(self, x):
        return paddle.isinf(x)

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestIsInfCase1(TestIsInfOp):
    def init_case(self):
        self.inputs = {"x": self.random([32, 64])}
        self.inputs["x"][0] = [np.inf] * 64


if __name__ == "__main__":
    unittest.main()
