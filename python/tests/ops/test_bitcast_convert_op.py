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
class TestBitcastConvertOp(OpTest):
    def setUp(self):
        self.init_case()

    # input[(3, 1), int32] --> output[(3, 1, 4), uint8]
    def init_case(self):
        self.inputs = {
            "x":
            np.array([
                [0],
                [65535],  # 00000000 00000000 11111111 11111111
                [255],  # 00000000 00000000 00000000 11111111
            ]).astype(np.int32)
        }
        self.outputs = {
            "y":
            np.array([
                [[0, 0, 0, 0]],
                [[255, 255, 0, 0]],
                [[255, 0, 0, 0]],
            ]).astype(np.uint8),
            "output_type":
            "uint8"
        }

    def build_paddle_program(self, target):
        y = paddle.to_tensor(self.outputs["y"], stop_gradient=False)
        self.paddle_outputs = [y]

    def build_cinn_program(self, target):
        builder = NetBuilder("bitcast_convert")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape, "x")
        out = builder.bitcast_convert(x, self.outputs["output_type"])
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]],
                                   [out])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestBitcastConvertCase1(TestBitcastConvertOp):
    # input[(4, 2), int16] --> output[(4), int32]
    def init_case(self):
        self.inputs = {
            "x":
            np.array([
                # 255: (00000000 11111111)
                [255, 255],
                [255, 255],
                [255, 255],
                [255, 255],
            ]).astype(np.int16)
        }
        self.outputs = {
            "y":
            np.array([
                # 16711935: (00000000 11111111 00000000 11111111)
                16711935,
                16711935,
                16711935,
                16711935
            ]).astype(np.int32),
            "output_type":
            "int32"
        }


class TestBitcastConvertCase2(TestBitcastConvertOp):
    # input[(4, 3, 2), float32] --> output[(4, 3), float64]
    def init_case(self):
        self.inputs = {
            "x":
            np.array([
                [[-2.4186804, 0.72310793], [-0.36041325, 0.32869506],
                 [-0.9628911, 0.5394769]],
                [[-0.96317023, -2.0704348], [1.4205104, -1.057216],
                 [-1.0708642, -0.19117601]],
                [[0.37250426, 0.1138144], [-0.44552958, 0.52443033],
                 [0.2424864, 0.9873532]],
                [[1.260338, -1.9353563], [-1.2953576, -0.27701086],
                 [0.07074322, 0.3671813]],
            ]).astype(np.float32)
        }
        self.outputs = {
            "y":
            np.array([
                [3.83234292e-04, 7.23954483e-07, 4.97934160e-05],
                [-2.56347990e+00, -1.13885049e-02, -9.20344510e-09],
                [1.82567544e-10, 4.24464741e-05, 7.02207626e-03],
                [-1.48285031e+00, -2.22247621e-07, 1.66874110e-06],
            ]).astype(np.float64),
            "output_type":
            "float64"
        }


class TestBitcastConvertCase3(TestBitcastConvertOp):
    # input[(4, 3, 2), float32] --> output[(4, 3, 2, 2), uint16]
    def init_case(self):
        self.inputs = {
            "x":
            np.array([
                [[-2.4186804, 0.72310793], [-0.36041325, 0.32869506],
                 [-0.9628911, 0.5394769]],
                [[-0.96317023, -2.0704348], [1.4205104, -1.057216],
                 [-1.0708642, -0.19117601]],
                [[0.37250426, 0.1138144], [-0.44552958, 0.52443033],
                 [0.2424864, 0.9873532]],
                [[1.260338, -1.9353563], [-1.2953576, -0.27701086],
                 [0.07074322, 0.3671813]],
            ]).astype(np.float32)
        }
        self.outputs = {
            "y":
            np.array([[[[52137, 49178], [7578, 16185]],
                       [[34838, 48824], [19128, 16040]],
                       [[32776, 49014], [6952, 16138]]],
                      [[[37459, 49014], [33281, 49156]],
                       [[54089, 16309], [21211, 49031]],
                       [[4628, 49033], [50085, 48707]]],
                      [[[47329, 16062], [6022, 15849]],
                       [[7284, 48868], [16657, 16134]],
                       [[20059, 15992], [49966, 16252]]],
                      [[[21185, 16289], [47553, 49143]],
                       [[52807, 49061], [54366, 48781]],
                       [[57810, 15760], [65328, 16059]]]]).astype(np.uint16),
            "output_type":
            "uint16"
        }


if __name__ == "__main__":
    unittest.main()
