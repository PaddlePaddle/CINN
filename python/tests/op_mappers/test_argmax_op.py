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

import sys
import unittest
import numpy as np
from op_mapper_test import OpMapperTest
import paddle
from cinn.frontend import *
from cinn.common import *

paddle.enable_static()


class TestArgmaxOp(OpMapperTest):
    def init_input_data(self):
        self.shape = [2, 3, 4]
        self.axis = None
        self.input_dtype = "float32"
        self.output_dtype = "int32"
        self.keepdim = False
        self.feed_data = {
            'x': self.random(self.shape, self.input_dtype),
        }

    def set_paddle_program(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype)
        out = paddle.argmax(x, self.axis, self.keepdim, self.output_dtype)

        return ([x.name], [out])

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestArgmaxCase1(TestArgmaxOp):
    """
    Test case with negative axis
    """

    def init_input_data(self):
        super().init_input_data()
        self.axis = -1


class TestArgmaxCase2(TestArgmaxOp):
    """
    Test case with unspecified axis and dtype
    """

    def init_input_data(self):
        super().init_input_data()
        self.axis = None
        self.output_dtype = "int32"
        self.keepdim = False


class TestArgmaxCase3(TestArgmaxOp):
    """
    Test case with dtype int32
    """

    def init_input_data(self):
        super().init_input_data()
        self.shape = [2, 3, 4]
        self.axis = 1
        self.input_dtype = "int32"
        self.feed_data = {
            'x': self.random(self.shape, self.input_dtype),
        }


class TestArgmaxCase4(TestArgmaxOp):
    """
    Test case with different input dtype and output dtype
    """

    def init_input_data(self):
        super().init_input_data()
        self.shape = [2, 3, 4]
        self.axis = 1
        self.input_dtype = "int16"
        self.keepdim = True
        self.feed_data = {
            'x': self.random(self.shape, self.input_dtype),
        }


class TestArgmaxCase5(TestArgmaxOp):
    """
    Test case with uint8 input 
    """

    def init_input_data(self):
        super().init_input_data()
        self.input_dtype = "uint8"
        self.feed_data = {
            'x': self.random(self.shape, self.input_dtype),
        }


class TestArgmaxCase6(TestArgmaxOp):
    """
    Test case with float64 input
    """

    def init_input_data(self):
        self.feed_data = {
            'x': self.random(self.shape, self.input_dtype),
        }
        self.input_dtype = "float64"
        self.feed_data = {
            'x': self.random(self.shape, self.input_dtype),
        }


if __name__ == "__main__":
    unittest.main()
