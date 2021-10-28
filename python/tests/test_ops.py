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

import paddle
import paddle.fluid as fluid
from cinn.frontend import *
from cinn import Target
from cinn.framework import *
import unittest
import cinn
from cinn import runtime
from cinn import ir
from cinn import lang
from cinn.common import *
import numpy as np
import paddle.fluid as fluid
import sys

enable_gpu = sys.argv.pop()


class TestReshape(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

    def paddle_verify(self, result):
        main_program = fluid.Program()
        paddle.enable_static()

        with fluid.program_guard(main_program, fluid.Program()):

            a = fluid.data(name='A', shape=[2, 2048, 1, 1], dtype='float32')
            res = paddle.reshape(x=a, shape=[-1, 2048])

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            x = np.array(result[0]).reshape((2, 2048, 1, 1)).astype("float32")
            output = exe.run(feed={"A": x}, fetch_list=[res])
            output = np.array(output).reshape(-1)
            print("result in paddle_verify: \n")
            for i in range(0, output.shape[0]):
                if np.abs(output[i] - result[len(result) - 1][i]) > 1e-4:
                    print("Error! ", i,
                          "-th data has diff with target data:\n", output[i],
                          " vs: ", result[len(result) - 1][i], ". Diff is: ",
                          output[i] - result[len(result) - 1][i])
            self.assertTrue(
                np.allclose(result[len(result) - 1], output, atol=1e-4))

    def test_basic(self):
        prog = Program()

        a = Variable("A").set_type(Float(32)).set_shape([2, 2048, 1, 1])
        h = prog.reshape(a, [-1, 2048])
        self.assertEqual(prog.size(), 1)
        # print program
        for i in range(prog.size()):
            print(prog[i])
        tensor_data = [np.random.random([2, 2048, 1, 1]).astype("float32")]
        result = prog.build_and_get_output(self.target, [a], tensor_data, [h])
        result = result[0].numpy(self.target).reshape(-1)
        tensor_data.append(result)
        self.paddle_verify(tensor_data)


class TestConcat(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

    def paddle_verify(self, result):
        main_program = fluid.Program()
        paddle.enable_static()

        with fluid.program_guard(main_program, fluid.Program()):

            a = fluid.data(name='A', shape=[1, 256, 14, 14], dtype='float32')
            b = fluid.data(name='B', shape=[1, 256, 14, 14], dtype='float32')
            c = fluid.data(name='C', shape=[1, 256, 14, 14], dtype='float32')
            res = fluid.layers.concat(input=[a, b, c], axis=1)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            x = np.array(result[0]).reshape((1, 256, 14, 14)).astype("float32")
            y = np.array(result[1]).reshape((1, 256, 14, 14)).astype("float32")
            z = np.array(result[2]).reshape((1, 256, 14, 14)).astype("float32")
            output = exe.run(feed={"A": x, "B": y, "C": z}, fetch_list=[res])
            output = np.array(output).reshape(-1)
            print("result in paddle_verify: \n")
            for i in range(0, output.shape[0]):
                if np.abs(output[i] - result[len(result) - 1][i]) > 1e-4:
                    print("Error! ", i,
                          "-th data has diff with target data:\n", output[i],
                          " vs: ", result[len(result) - 1][i], ". Diff is: ",
                          output[i] - result[len(result) - 1][i])
            self.assertTrue(
                np.allclose(result[len(result) - 1], output, atol=1e-4))

    def test_basic(self):
        prog = Program()

        a = Variable("A").set_type(Float(32)).set_shape([1, 256, 14, 14])
        b = Variable("B").set_type(Float(32)).set_shape([1, 256, 14, 14])
        c = Variable("C").set_type(Float(32)).set_shape([1, 256, 14, 14])
        h = prog.concat([a, b, c], 1)
        self.assertEqual(prog.size(), 1)
        # print program
        for i in range(prog.size()):
            print(prog[i])
        tensor_data = [
            np.random.random([1, 256, 14, 14]).astype("float32"),
            np.random.random([1, 256, 14, 14]).astype("float32"),
            np.random.random([1, 256, 14, 14]).astype("float32")
        ]
        result = prog.build_and_get_output(self.target, [a, b, c], tensor_data,
                                           [h])
        result = result[0].numpy(self.target).reshape(-1)
        tensor_data.append(result)
        self.paddle_verify(tensor_data)


if __name__ == "__main__":
    unittest.main()
