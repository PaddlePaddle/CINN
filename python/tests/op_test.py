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
from cinn import Target
from cinn.frontend import *
from cinn.common import *
import numpy as np
import paddle


class OpTest(unittest.TestCase):
    def init_results(self):
        self.paddle_outputs = []
        self.paddle_grads = []
        self.cinn_outputs = []
        self.cinn_grads = []

    def init_target(self):
        self.targets = [
            DefaultHostTarget(),
            DefaultNVGPUTarget(),
        ]

    def build_paddle_program():
        raise Exception("Not implemented.")

    def get_paddle_grads(self, outputs, inputs):
        grads = paddle.grad(outputs, inputs)
        return grads

    def build_cinn_program():
        raise Exception("Not implemented.")

    def get_cinn_output(self, prog, inputs, feed_data, outputs):
        self.apply_pass(prog, self.targets[1])
        result = prog.build_and_get_output(self.targets[1], inputs, feed_data,
                                           outputs)
        outs_and_grads = []
        for res in result:
            outs_and_grads.append(res.numpy(self.targets[1]))

        return outs_and_grads

    def apply_pass(self, prog, target):
        print("============ Before Decomposer Pass ============")
        for i in range(prog.size()):
            print(prog[i])

        prog.apply_pass(target, ["Decomposer"])
        print("============ After Decomposer Pass ============")
        for i in range(prog.size()):
            print(prog[i])

    def check_outputs_and_grads(self):
        self.build_paddle_program()
        self.build_cinn_program()
        print("============ Check Outputs ============")
        self.check_results(self.paddle_outputs, self.cinn_outputs)
        print("============ Check Grads ============")
        self.check_results(self.paddle_grads, self.cinn_grads)

    def check_results(self, expect_res, actual_res):
        print(len(expect_res), len(actual_res))
        self.assertEqual(len(expect_res), len(actual_res))
        for i in range(len(expect_res)):
            print("Check the %d -th Result..." % i)
            self.assertTrue(
                np.allclose(expect_res[i], actual_res[i], atol=1e-4))
