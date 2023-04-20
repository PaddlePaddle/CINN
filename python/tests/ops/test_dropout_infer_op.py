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
import numpy as np
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper
import paddle
import cinn
from cinn.frontend import *
from cinn.common import *


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestDropoutInferOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.case["x_shape"], dtype=self.case["x_dtype"])
        if self.case["mode"] is 'upscale_in_train':
            self.case["cinn_mode"] = 'upscale_in_train'
        elif self.case["mode"] is 'downscale_in_infer':
            self.case["cinn_mode"] = 'downgrade_in_infer'
        else:
            raise f"Unknown mode for dropout_infer: {self.case['mode']}"

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        out = paddle.nn.functional.dropout(
            x, p=self.case["p"], mode=self.case["mode"], training=False)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("dropout_infer")
        x = builder.create_input(
            self.nptype2cinntype(self.case["x_dtype"]), self.case["x_shape"],
            "x")
        out = builder.dropout_infer(x, self.case["p"], self.case["cinn_mode"])

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.x_np], [out])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        max_relative_error = self.case[
            "max_relative_error"] if "max_relative_error" in self.case else 1e-5
        self.check_outputs_and_grads(max_relative_error=max_relative_error)


class TestDropoutInferAll(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestDropoutInferOpCase"
        self.cls = TestDropoutInferOp
        self.inputs = [{
            "x_shape": [1024],
        }, {
            "x_shape": [32, 64],
        }, {
            "x_shape": [3, 32, 64],
        }, {
            "x_shape": [1, 32, 32, 3],
        }]
        self.dtypes = [
            {
                "x_dtype": "float32",
            },
            {
                "x_dtype": "float64",
            },
        ]
        self.attrs = {
            "p": [0.1, 0.5],
            "mode": ['upscale_in_train', 'downscale_in_infer'],
        }


if __name__ == "__main__":
    TestDropoutInferAll().run()
