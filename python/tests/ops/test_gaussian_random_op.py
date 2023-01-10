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
            np.array([[0.27076274, -0.24768609, -0.7465344],
                      [-0.5368036, -0.8755634, 0.618268]]).astype(np.float32)
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
            np.array([[[1.5415255, 0.5046278, -0.4930688, -0.07360721],
                       [-0.75112677, 2.236536, 1.4920039, 2.981348],
                       [3.4317055, -1.1959467, 0.4004395, 0.16091263]],
                      [[-1.1579778, 0.764502, 2.3821187, 0.08788824],
                       [2.5972023, -1.1975739, -1.0235088, 2.3628762],
                       [1.1202344, 1.0854168, 1.7805797,
                        0.05267286]]]).astype(np.float32)
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
            np.array([[[-1.80359845, 0.12100626, 0.67122622, 4.44270347],
                       [3.5119238, 2.82760506, 5.3521794, -4.87603233],
                       [1.14078774, 6.16050423, 5.03491447, 2.14809939]],
                      [[4.28213893, 2.6246409, 6.34640462, 1.385498],
                       [0.16001593, 5.62062184, 1.54228776, 3.68409538],
                       [-3.06390444, -0.88149752, 4.28376703,
                        5.4233008]]]).astype(np.float64)
        }
        self.shape = [2, 3, 4]
        self.mean = 2.0
        self.std = 3.0
        self.seed = 10
        self.dtype = "float64"


if __name__ == "__main__":
    unittest.main()
