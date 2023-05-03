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
from op_test import OpTest, OpTestTool
import paddle
from cinn.frontend import *
from cinn.common import *


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestIdentityOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {
            "x": self.random(self.init_shape(), self.init_dtype()),
        }

    def init_dtype(self):
        return 'float32'

    def init_shape(self):
        return [100, 100]

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs['x'], stop_gradient=True)
        out = paddle.assign(x)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("identity")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape, "x")
        out = builder.identity(x)

        prog = builder.build()

        res = self.get_cinn_output(
            prog, target, [x], [self.inputs["x"]], [out], passes=[])

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestIdentityFP64(TestIdentityOp):
    def init_dtype(self):
        return 'float32'


class TestIdentityFP16(TestIdentityOp):
    def init_dtype(self):
        return 'float16'


class TestIdentityInt32(TestIdentityOp):
    def init_dtype(self):
        return 'int32'


class TestIdentityInt64(TestIdentityOp):
    def init_dtype(self):
        return 'int64'


class TestIdentityInt8(TestIdentityOp):
    def init_dtype(self):
        return 'int8'


class TestIdentityInt16(TestIdentityOp):
    def init_dtype(self):
        return 'int16'


class TestIdentityBool(TestIdentityOp):
    def init_dtype(self):
        return 'bool'


if __name__ == "__main__":
    unittest.main()
