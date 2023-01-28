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
class TestUniformRandomOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.outputs = {
            "out":
            np.array([[0.96705663, 0.19087338, 0.0266993],
                      [-0.6569866, -0.51567507,
                       -0.4805799]]).astype(np.float32)
        }
        self.shape = [2, 3]
        self.min = -1.0
        self.max = 1.0
        self.seed = 10
        self.dtype = "float32"

    def build_paddle_program(self, target):
        out = paddle.to_tensor(self.outputs["out"], stop_gradient=True)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("uniform_random")
        out = builder.uniform_random(self.shape, self.min, self.max, self.seed,
                                     self.dtype)
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [], [], [out], passes=[])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestUniformRandomCase1(TestUniformRandomOp):
    def init_case(self):
        self.outputs = {
            "out":
            np.array([[[5.3188114, 1.0498036, 0.14684618, -3.6134262],
                       [-2.836213, -2.6431894, 2.0782926, 2.870666],
                       [3.1589284, 0.70414567, 4.580574, 4.3410697]],
                      [[-4.4212003, 5.053883, -4.7842636, 1.6527312],
                       [1.4486676, 4.8414164, -0.9332043, -3.404669],
                       [-2.8361645, -0.35731608, 4.514214,
                        -2.6455283]]]).astype(np.float32)
        }
        self.shape = [2, 3, 4]
        self.min = -5.5
        self.max = 5.5
        self.seed = 10
        self.dtype = "float32"


class TestUniformRandomCase2(TestUniformRandomOp):
    def init_case(self):
        self.outputs = {
            "out":
            np.array([[[1.90873906, -6.56987078, 0.96422553, 3.16367954],
                       [-4.80580041, 5.2193862, -3.76984434, 0.45475599],
                       [1.28026275, 7.89284877, 7.32952979, 3.30392344]],
                      [[9.18887966, 3.00496491, -3.31859432, 4.23467543],
                       [8.80256917, -6.19030602, -7.62713425, 4.40370033],
                       [-0.64966534, -4.81005288, -1.38839721,
                        -0.29885071]]]).astype(np.float64)
        }
        self.shape = [2, 3, 4]
        self.min = -10.0
        self.max = 10.0
        self.seed = 10
        self.dtype = "float64"


if __name__ == "__main__":
    unittest.main()
