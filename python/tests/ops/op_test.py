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
logger = logging.getLogger(name="op_test")


class OpTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(OpTest, self).__init__(*args, **kwargs)
        self._init_target()
        self._init_results()

    def _init_results(self):
        self.paddle_outputs = []
        self.paddle_grads = []
        self.cinn_outputs = []
        self.cinn_grads = []

    def _init_target(self):
        self.target = DefaultHostTarget()
        if is_compiled_with_cuda():
            self.target = DefaultNVGPUTarget()

    def _get_device(self):
        return "NVGPU" if is_compiled_with_cuda() else "CPU"

    def build_paddle_program(self, target):
        raise Exception("Not implemented.")

    def get_paddle_grads(self, outputs, inputs, grad_outputs):
        grad_tensors = []
        for grad in grad_outputs:
            grad_tensors.append(paddle.to_tensor(grad))
        grads = paddle.grad(outputs, inputs, grad_tensors)

        return grads

    def build_cinn_program(self, target):
        raise Exception("Not implemented.")

    def get_cinn_output(self,
                        prog,
                        target,
                        inputs,
                        feed_data,
                        outputs,
                        passes=[],
                        scope=None):
        fetch_ids = {str(out) for out in outputs}
        result = prog.build_and_get_output(
            target, inputs, feed_data, outputs, passes=passes, scope=scope)
        outs_and_grads = []
        for res in result:
            outs_and_grads.append(res.numpy(target))

        return outs_and_grads

    def apply_pass(self, prog, target, passes=["Decomposer"], fetch_ids=set()):
        def print_program(prog):
            if logger.getEffectiveLevel() != logging.DEBUG:
                return
            for i in range(prog.size()):
                print(prog[i])

        logger.debug("============ Before Decomposer Pass ============")
        print_program(prog)

        prog.apply_pass(fetch_ids, target, passes)

        logger.debug("============ After Decomposer Pass ============")
        print_program(prog)

    def check_outputs_and_grads(self,
                                max_relative_error=1e-5,
                                all_equal=False,
                                equal_nan=False):
        self.build_paddle_program(self.target)
        self.build_cinn_program(self.target)

        logger.debug("============ Check Outputs ============")
        self.check_results(self.paddle_outputs, self.cinn_outputs,
                           max_relative_error, all_equal, equal_nan, "Outputs")

        if len(self.cinn_grads) != 0:
            logger.debug("============ Check Grads ============")
            self.check_results(self.paddle_grads, self.cinn_grads,
                               max_relative_error, all_equal, equal_nan,
                               "Grads")

    def check_results(self,
                      expect_res,
                      actual_res,
                      max_relative_error,
                      all_equal=False,
                      equal_nan=False,
                      name="Outputs"):
        def _compute_max_relative_error(output_id, expect, actual):
            absolute_diff = np.abs(expect - actual).flatten()
            relative_diff = absolute_diff / np.abs(expect).flatten()
            max_diff = 0
            min_diff = max_relative_error
            offset = 0
            num_diffs = 0
            for i in range(len(relative_diff)):
                if relative_diff[i] > max_diff:
                    max_diff = relative_diff[i]
                    offset = i
                if relative_diff[i] > max_relative_error:
                    num_diffs = num_diffs + 1
                    # The following print can be used to debug.
                    # print("i=%d, %e vs %e, relative_diff=%e, absolute_diff=%e" % (i, expect.flatten()[i], actual.flatten()[i], relative_diff[i], absolute_diff[i]))
            error_message = "[%s] The %d-th output: total %d different results, offset=%d, shape=%s, %e vs %e, maximum_relative_diff=%e (absolute_diff=%e)." % (
                self._get_device(), output_id, num_diffs, offset,
                str(expect.shape), expect.flatten()[offset],
                actual.flatten()[offset], max_diff, absolute_diff[offset])
            return error_message

        def _check_error_message(output_id, expect, actual):
            expect_flatten = expect.flatten()
            actual_flatten = actual.flatten()
            self.assertEqual(
                len(expect_flatten), len(actual_flatten),
                "[{}] The {}-th output size different, which expect shape is {} but actual is {}."
                .format(self._get_device(), output_id, expect.shape,
                        actual.shape))
            num_diffs = 0
            offset = -1
            for i in range(len(expect_flatten)):
                if expect_flatten[i] != actual_flatten[i]:
                    num_diffs = num_diffs + 1
                    offset = i if offset == -1 else offset

            error_message = "[{}] The {}-th output: total {} different results, the first different result's offset={}, where expect value is {} but actual is {}.".format(
                self._get_device(), output_id, num_diffs, offset,
                expect_flatten[offset], actual_flatten[offset])
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

            self.assertEqual(
                expect.dtype,
                actual.dtype,
                msg=
                "[{}] The {}-th output dtype different, which expect shape is {} but actual is {}."
                .format(self._get_device(), i, expect.dtype, actual.dtype))
            self.assertEqual(
                expect.shape,
                actual.shape,
                msg=
                "[{}] The {}-th output shape different, which expect shape is {} but actual is {}."
                .format(self._get_device(), i, expect.shape, actual.shape))

            should_all_equal = all_equal or (actual.dtype in [
                np.dtype('bool'),
                np.dtype('int32'),
                np.dtype('int64')
            ])

            is_allclose = True
            error_message = ""
            if not should_all_equal:
                is_allclose = np.allclose(
                    expect,
                    actual,
                    atol=1e-6,
                    rtol=max_relative_error,
                    equal_nan=equal_nan)
                error_message = "np.allclose(expect, actual, atol=1e-6, rtol={}) checks succeed!".format(
                    max_relative_error
                ) if is_allclose else _compute_max_relative_error(
                    i, expect, actual)
            else:
                is_allclose = np.all(expect == actual)
                error_message = "(expect == actual) checks succeed!" if is_allclose else _check_error_message(
                    i, expect, actual)

            error_message = "[Check " + name + "] " + error_message

            logger.debug("{} {}".format(is_allclose, error_message))
            self.assertTrue(is_allclose, msg=error_message)

    @staticmethod
    def paddleddtype2str(dtype):
        switch_map = {
            paddle.float16: "float16",
            paddle.float32: "float32",
            paddle.float64: "float64",
            paddle.int8: "int8",
            paddle.int16: "int16",
            paddle.int32: "int32",
            paddle.int64: "int64",
            paddle.uint8: "uint8",
            paddle.bool: "bool",
        }
        assert dtype in switch_map, str(dtype) + " not support in CINN"
        return switch_map[dtype]

    @staticmethod
    def nptype2cinntype(dtype):
        switch_map = {
            "float16": Float(16),
            "float32": Float(32),
            "float64": Float(64),
            "int8": Int(8),
            "int16": Int(16),
            "int32": Int(32),
            "int64": Int(64),
            "uint8": UInt(8),
            "uint16": UInt(16),
            "uint32": UInt(32),
            "uint64": UInt(64),
            "bool": Bool()
        }
        assert str(dtype) in switch_map, str(dtype) + " not support in CINN"
        return switch_map[str(dtype)]

    @staticmethod
    def paddleddtype2cinntype(dtype):
        return OpTest.nptype2cinntype(OpTest.paddleddtype2str(dtype))

    @staticmethod
    def random(shape, dtype="float32", low=0.0, high=1.0):
        assert bool(shape), "Shape should not empty!"
        assert -1 not in shape, "Shape should not -1!"
        if dtype in ["float16", "float32", "float64"]:
            return np.random.uniform(low, high, shape).astype(dtype)
        elif dtype == "bool":
            return np.random.choice(a=[False, True], size=shape).astype(dtype)
        elif dtype in ["int8", "uint8", "int32", "int64"]:
            return np.random.randint(low, high, shape).astype(dtype)
        else:
            raise Exception("Not supported yet.")


class OpTestTool:
    @classmethod
    def skip_if(cls, condition: object, reason: str):
        return unittest.skipIf(condition, reason)
