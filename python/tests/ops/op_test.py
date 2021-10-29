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
import logging
from contextlib import contextmanager
import os

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO').upper())
logger = logging.getLogger()


class OpTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(OpTest, self).__init__(*args, **kwargs)
        self.init_target()
        self.init_results()

    def init_results(self):
        self.paddle_outputs = []
        self.paddle_grads = []
        self.cinn_outputs = []
        self.cinn_grads = []

    def init_target(self):
        self.targets = [DefaultHostTarget()]
        if is_compiled_with_cuda():
            self.targets.append(DefaultNVGPUTarget())

    def build_paddle_program():
        raise Exception("Not implemented.")

    def get_paddle_grads(self, outputs, inputs, grad_outputs):
        grad_tensors = []
        for grad in grad_outputs:
            grad_tensors.append(paddle.to_tensor(grad))
        grads = paddle.grad(outputs, inputs, grad_tensors)

        return grads

    def build_cinn_program():
        raise Exception("Not implemented.")

    def get_cinn_output(self, prog, target, inputs, feed_data, outputs):
        self.apply_pass(prog, target)
        result = prog.build_and_get_output(target, inputs, feed_data, outputs)
        outs_and_grads = []
        for res in result:
            outs_and_grads.append(res.numpy(target))

        return outs_and_grads

    def apply_pass(self, prog, target, passes=["Decomposer"]):
        def print_program(prog):
            if logger.getEffectiveLevel() != logging.DEBUG:
                return
            for i in range(prog.size()):
                print(prog[i])

        logger.debug("============ Before Decomposer Pass ============")
        print_program(prog)

        prog.apply_pass(target, passes)

        logger.debug("============ After Decomposer Pass ============")
        print_program(prog)

    def check_outputs_and_grads(self):
        self.build_paddle_program(self.targets[0])
        for target in self.targets:
            self.build_cinn_program(target)
            logger.debug("============ Check Outputs ============")
            self.check_results(self.paddle_outputs, self.cinn_outputs)

            if len(self.cinn_grads) != 0:
                logger.debug("============ Check Grads ============")
                self.check_results(self.paddle_grads, self.cinn_grads)

    def check_results(self, expect_res, actual_res):
        def _max_relative_error(i, expect, actual, max_relative_error=1e-5):
            diff = (np.abs(expect - actual) / np.abs(expect)).flatten()
            max_diff = np.max(diff)
            offset = np.argmax(diff > max_relative_error)
            error_message = "The maximum relative error of %d-th output is %e, offset=%d, shape=%s, %e vs %e." % (
                i, max_diff, offset, str(expect.shape),
                expect.flatten()[offset], actual.flatten()[offset])
            return error_message

        self.assertEqual(len(expect_res), len(actual_res))
        for i in range(len(expect_res)):
            if expect_res[i] is None:
                continue

            if isinstance(expect_res[i], paddle.Tensor):
                expect = expect_res[i].numpy()
            else:
                expect = expect_res[i]
            actual = actual_res[i]
            is_allclose = np.allclose(expect, actual, atol=1e-6)
            self.assertTrue(is_allclose, _max_relative_error(
                i, expect, actual))


class OpTestTool:
    @classmethod
    def skip_if(cls, condition: object, reason: str):
        return unittest.skipIf(condition, reason)
