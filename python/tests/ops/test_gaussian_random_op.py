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
import cinn
from cinn.frontend import *
from cinn.common import *


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestGaussianRandomOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.outputs = {
            "out":
            np.array([[-0.10285693, -0.15045978, 1.017193],
                      [0.54673094, 1.6809963, -0.1026846]]).astype(np.float32)
        }
        self.shape = [2, 3]
        self.mean = 0.0
        self.std = 1.0
        self.seed = 10
        self.dtype = "float32"

    def build_paddle_program(self, target):
        out = paddle.to_tensor(self.outputs["out"], stop_gradient=True)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("gaussian_random")
        out = builder.gaussian_random(self.shape, self.mean, self.std,
                                      self.seed, self.dtype)
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [], [], [out], passes=[])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestGaussianRandomCase1(TestGaussianRandomOp):
    def init_case(self):
        self.outputs = {
            "out":
            np.array([[[0.79428613, 0.69908047, 3.034386, 2.093462],
                       [4.361993, 0.79463077, -0.722417, 1.1189039],
                       [0.4583773, -0.273216, 0.48635447, 1.6591272]],
                      [[-0.0864414, 5.1708527, -2.7867026, -1.7421858],
                       [0.29575956, 2.782921, 3.4686515, 1.9687234],
                       [1.6826894, -2.2983267, 1.865128,
                        0.94831306]]]).astype(np.float32)
        }
        self.shape = [2, 3, 4]
        self.mean = 1.0
        self.std = 2.0
        self.seed = 10
        self.dtype = "float32"


class TestGaussianRandomCase2(TestGaussianRandomOp):
    def init_case(self):
        self.outputs = {
            "out":
            np.array([[[4.6908195, 3.44629396, -0.75698307, 0.20591365],
                       [-2.91450108, 2.33925561, 1.34762535, -2.53523588],
                       [0.02648256, 4.5324747, 0.61649971, 1.18410024]],
                      [[1.30081438, 1.49367627, -2.31465181, 0.94214862],
                       [2.98133001, 2.38508311, -4.08593164, 0.84637714],
                       [5.69284255, 1.77937267, 2.36510564,
                        -1.87735613]]]).astype(np.float64)
        }
        self.shape = [2, 3, 4]
        self.mean = 2.0
        self.std = 3.0
        self.seed = 10
        self.dtype = "float64"


if __name__ == "__main__":
    unittest.main()
