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


class TestNetBuilder(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

    def test_basic(self):
        builder = NetBuilder("test_basic")
        a = builder.create_input(Float(32), (512, 32), "A")
        d = builder.relu(a)
        prog = builder.build()
        self.assertEqual(prog.size(), 1)
        # print program
        for i in range(prog.size()):
            print(prog[i])
        tensor_data = [np.random.random([512, 32]).astype("float32")]
        result = prog.build_and_get_output(self.target, [a], tensor_data, [d])


class TestNetBuilder2(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

    def test_basic(self):
        builder = NetBuilder("test_basic")
        a = builder.create_input(Float(32), (32, 32), "A")
        b = builder.create_input(Float(32), (32, 32), "B")
        d = builder.mul(a, b)
        prog = builder.build()
        self.assertEqual(prog.size(), 1)
        # print program
        for i in range(prog.size()):
            print(prog[i])
        tensor_data = [
            np.random.random([32, 32]).astype("float32"),
            np.random.random([32, 32]).astype("float32")
        ]
        result = prog.build_and_get_output(self.target, [a, b], tensor_data,
                                           [d])


class TestNetBuilder3(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

    def test_basic(self):
        builder = NetBuilder("test_basic")
        a = builder.create_input(Float(32), (512, 32), "A")
        d = builder.softmax(a)
        prog = builder.build()
        self.assertEqual(prog.size(), 1)
        # print program
        for i in range(prog.size()):
            print(prog[i])
        tensor_data = [np.random.random([512, 32]).astype("float32")]
        result = prog.build_and_get_output(self.target, [a], tensor_data, [d])


if __name__ == "__main__":
    unittest.main()
