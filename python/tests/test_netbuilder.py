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

enable_cudnn = sys.argv.pop()
enable_gpu = sys.argv.pop()


class TestNetBuilder(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

    def paddle_verify_basic(self, result):
        paddle.enable_static()

        a = fluid.layers.data(name='A', shape=[24, 56, 56], dtype='float32')
        b = fluid.layers.data(name='B', shape=[24, 56, 56], dtype='float32')
        c = fluid.layers.elementwise_add(a, b)
        d = fluid.initializer.NumpyArrayInitializer(
            np.array(result[2]).reshape((144, 24, 1, 1)).astype('float32'))
        res = fluid.layers.conv2d(
            input=c,
            num_filters=144,
            filter_size=1,
            stride=1,
            padding=0,
            dilation=1,
            param_attr=d)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        x = np.array(result[0]).reshape((1, 24, 56, 56)).astype("float32")
        y = np.array(result[1]).reshape((1, 24, 56, 56)).astype("float32")
        output = exe.run(feed={"A": x, "B": y}, fetch_list=[res])
        output = np.array(output).reshape(-1)
        print("result in paddle_verify: \n")
        for i in range(0, output.shape[0]):
            if np.abs(output[i] - result[len(result) - 1][i]) > 1e-4:
                print("Error! ", i, "-th data has diff with target data:\n",
                      output[i], " vs: ", result[len(result) - 1][i],
                      ". Diff is: ", output[i] - result[len(result) - 1][i])
        self.assertTrue(
            np.allclose(result[len(result) - 1], output, atol=1e-4))

    def test_basic(self):
        builder = NetBuilder("test_basic")
        a = builder.create_input(Float(32), (1, 24, 56, 56), "A")
        b = builder.create_input(Float(32), (1, 24, 56, 56), "B")
        c = builder.add(a, b)
        d = builder.create_input(Float(32), (144, 24, 1, 1), "D")
        e = builder.conv2d(c, d)
        prog = builder.build()
        self.assertEqual(prog.size(), 2)
        # print program
        for i in range(prog.size()):
            print(prog[i])
        tensor_data = [
            np.random.random([1, 24, 56, 56]).astype("float32"),
            np.random.random([1, 24, 56, 56]).astype("float32"),
            np.random.random([144, 24, 1, 1]).astype("float32")
        ]
        result = prog.build_and_get_output(self.target, [a, b, d], tensor_data,
                                           [e])
        result = result[0].numpy(self.target).reshape(-1)
        tensor_data.append(result)
        self.paddle_verify_basic(tensor_data)

    def test_mul(self):
        builder = NetBuilder("test_basic")
        a = builder.create_input(Float(32), (16, 64), "A")
        b = builder.create_input(Float(32), (16, 64), "B")
        c = builder.mul(a, b)
        prog = builder.build()
        self.assertEqual(prog.size(), 1)
        # print program
        for i in range(prog.size()):
            print(prog[i])
        tensor_data = [
            np.random.random([16, 64]).astype("float32"),
            np.random.random([16, 64]).astype("float32")
        ]
        result = prog.build_and_get_output(self.target, [a, b], tensor_data,
                                           [c])
        result = result[0].numpy(self.target)

        #numpy mul need transpose before
        if enable_gpu == "OFF":
            tensor_data[1] = np.transpose(tensor_data[1])
        elif enable_cudnn == "OFF":
            tensor_data[1] = np.transpose(tensor_data[1])
        tensor_data[1] = tensor_data[1].reshape(64, 16)
        c_np = np.matmul(tensor_data[0], tensor_data[1])
        self.assertTrue(np.allclose(result, c_np, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
