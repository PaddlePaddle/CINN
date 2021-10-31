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

assert len(sys.argv) == 1 + 2 + 1  # model and enable_gpu count
enable_gpu = sys.argv.pop()
multi_fc_model_dir = sys.argv.pop()
naive_model_dir = sys.argv.pop()


class TestFrontend(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

    def paddle_verify(self, result):
        paddle.enable_static()

        a = fluid.layers.data(name='A', shape=[512, 7, 7], dtype='float32')
        e = fluid.initializer.NumpyArrayInitializer(
            np.array(result[1]).reshape((512, 512, 3, 3)).astype("float32"))
        f = fluid.layers.conv2d(
            input=a,
            num_filters=512,
            filter_size=3,
            stride=1,
            padding=1,
            dilation=1,
            param_attr=e)
        res = fluid.layers.relu(f)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        x = np.array(result[0]).reshape((1, 512, 7, 7)).astype("float32")
        output = exe.run(feed={"A": x}, fetch_list=[res])
        output = np.array(output).reshape(-1)
        print("result in paddle_verify: \n")
        nums = 0
        for i in range(0, 20):
            if np.abs(output[i] - result[len(result) - 1][i]) > 1e-2:
                print("Error! ", i, "-th data has diff with target data:\n",
                      output[i], " vs: ", result[len(result) - 1][i],
                      ". Diff is: ", output[i] - result[len(result) - 1][i])
        for i in range(0, output.shape[0]):
            if np.abs(output[i] - result[len(result) - 1][i]) > 1e-2:
                nums = nums + 1
        print("total different num is :", nums)
        print(result[len(result) - 1])
        print(output)
        self.assertTrue(
            np.allclose(result[len(result) - 1], output, rtol=0, atol=1e-2))

    def test_basic(self):
        prog = Program()

        a = Variable("A").set_type(Float(32)).set_shape([1, 512, 7, 7])
        e = Variable("E").set_type(Float(32)).set_shape([512, 512, 3, 3])
        f = prog.conv2d(a, e, {
            "stride": [1, 1],
            "dilation": [1, 1],
            "padding": [1, 1]
        })
        h = prog.relu(f)

        self.assertEqual(prog.size(), 2)
        # print program
        for i in range(prog.size()):
            print(prog[i])
        tensor_data = [
            np.random.random([1, 512, 7, 7]).astype("float32"),
            np.random.random([512, 512, 3, 3]).astype("float32")
            # np.ones([1, 512, 7, 7]).astype("float32") /2,
            # np.ones([512, 512, 3, 3]).astype("float32") /2
        ]
        result = prog.build_and_get_output(self.target, [a, e], tensor_data, h)
        result = result.numpy(self.target).reshape(-1)
        tensor_data.append(result)
        self.paddle_verify(tensor_data)


if __name__ == "__main__":
    unittest.main()
