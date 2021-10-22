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


class OpTest(unittest.TestCase):
    def init_target(self):
        self.targets = [
            DefaultHostTarget(),
            DefaultNVGPUTarget(),
        ]

    def build_paddle_program():
        pass

    def get_paddle_grads(self, outputs, inputs):
        grads = paddle.grad(outputs, inputs)
        return grads

    def build_cinn_program():
        pass

    def get_cinn_output(self, prog, inputs, feed_data, outputs):
        print("============ Before Decomposer Pass ============")
        for i in range(prog.size()):
            print(prog[i])

        prog.apply_pass(self.targets[0], ["Decomposer"])
        print("============ Before Decomposer Pass ============")
        for i in range(prog.size()):
            print(prog[i])

        result = prog.build_and_get_output(self.targets[0], inputs, feed_data,
                                           outputs)
        result = result.numpy(self.targets[0])

        return result

    def get_cinn_grads(self, outputs, inputs):
        pass

    def check_output(self):
        paddle_outs = self.build_paddle_program()
        cinn_outs = self.build_cinn_program()

        self.assertEqual(len(paddle_outs), len(cinn_outs))
        for i in range(len(paddle_outs)):
            print("Check the %d -th output..." % i)
            self.assertTrue(
                np.allclose(paddle_outs[i], cinn_outs[i], atol=1e-4))

    def check_grad(self, outputs, inputs):
        paddle_grads = self.get_paddle_grads()
        cinn_grads = self.get_cinn_grads()
