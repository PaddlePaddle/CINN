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
from pass_test import PassTest
from cinn.frontend import *
from cinn.common import *


class TestSimpleRecomputePass(PassTest):
    def init_input_data(self):
        self.feed_data = [self.random([4, 5, 6], "float16")]

    def build_program(self, builder, target):
        x = builder.create_input(
            self.nptype2cinntype(self.feed_data[0].dtype),
            self.feed_data[0].shape, "x")
        y = builder.cast(x, "float32")
        z1 = builder.exp(y)
        z2 = builder.log(y)
        z3 = builder.sqrt(y)
        out = builder.sum([z1, z2, z3])
        return [x], [out]

    def test_check_results(self):
        self.check_pass_outputs(
            pass_diff=-2, test_passes=["SimpleRecomputePass"])


class TestSimpleRecomputePassCase1(PassTest):
    def init_input_data(self):
        self.num_channels = 16
        self.feed_data = [self.random([2, self.num_channels, 8, 8], "float16")]

    def build_program(self, builder, target):
        x = builder.create_input(
            self.nptype2cinntype(self.feed_data[0].dtype),
            self.feed_data[0].shape, "x")

        scale = builder.fill_constant([self.num_channels], 1.0, 'scale',
                                      'float32')
        bias = builder.fill_constant([self.num_channels], 0.0, 'bias',
                                     'float32')
        mean = builder.fill_constant([self.num_channels], 0.0, 'mean',
                                     'float32')
        variance = builder.fill_constant([self.num_channels], 1.0, 'variance',
                                         'float32')

        out = builder.batchnorm(x, scale, bias, mean, variance, is_test=False)
        return [x], out

    def test_check_results(self):
        self.check_pass_outputs(
            pass_diff=-3, test_passes=["SimpleRecomputePass"])


class TestSimpleRecomputePassCase2(PassTest):
    def init_input_data(self):
        self.shape = [4, 5, 6]
        self.feed_data = [
            self.random(self.shape, "float32"),
            self.random(self.shape, "float32"),
        ]

    def build_program(self, builder, target):
        x1 = builder.create_input(
            self.nptype2cinntype(self.feed_data[0].dtype),
            self.feed_data[0].shape, "x1")
        x2 = builder.create_input(
            self.nptype2cinntype(self.feed_data[1].dtype),
            self.feed_data[0].shape, "x2")

        y = builder.fill_constant(self.shape, 1.0, 'y',
                                  str(self.feed_data[0].dtype))
        z1 = builder.add(x1, y)
        z2 = builder.subtract(x2, y)
        z3 = builder.exp(y)
        out = builder.sum([z1, z2, z3])

        return [x1, x2], [out]

    def test_check_results(self):
        self.check_pass_outputs(
            pass_diff=-2, test_passes=["SimpleRecomputePass"])


class TestSimpleRecomputePassCase3(PassTest):
    def init_input_data(self):
        self.shape = [4, 5, 6]
        self.feed_data = [
            self.random(self.shape, "float16"),
        ]

    def build_program(self, builder, target):
        x = builder.create_input(
            self.nptype2cinntype(self.feed_data[0].dtype),
            self.feed_data[0].shape, "x")
        y = builder.fill_constant(self.shape, 1.0, 'y', "float32")

        x_fp32 = builder.cast(x, "float32")
        z1 = builder.add(x_fp32, y)
        z2 = builder.subtract(x_fp32, y)
        z3 = builder.multiply(x_fp32, y)
        out = builder.sum([z1, z2, z3])

        return [x], [out]

    def test_check_results(self):
        self.check_pass_outputs(
            pass_diff=-4, test_passes=["SimpleRecomputePass"])


if __name__ == "__main__":
    unittest.main()
