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
        self.feed_data = {
            'x': np.array([1, 2, 3], dtype='float32'),
        }
        self.axis = None
        self.shape = [2, 3, 4]
        self.output_dtype = "int64"
        self.keepdim = False

    def set_op_type(self):
        return "arg_max"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype)
        return {'X': [x]}

    def set_op_attrs(self):
        return {
            "axis": self.axis,
            "keepdim": self.keepdim,
            "dtype": self.output_dtype
        }

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestArgmaxCase1(TestArgmaxOp):
    """
    Test case with negative axis
    """

    def init_input_data(self):
        self.feed_data = {
            'x': np.array([1, 2, 3], dtype='float32'),
        }
        self.axis = -1
        self.output_dtype = "int64"
        self.keepdim = False


class TestArgmaxCase2(TestArgmaxOp):
    """
    Test case with keepdim = true
    """

    def init_input_data(self):
        self.feed_data = {
            'x': np.array([1, 2, 3], dtype='float32'),
        }
        self.axis = None
        self.output_dtype = "int32"
        self.keepdim = True


class TestArgmaxCase4(TestArgmaxOp):
    """
    Test case with intput_dtype=uint8
    """

    def init_input_data(self):
        self.feed_data = {
            'x': np.array([1, 2, 3], dtype='float32'),
        }
        self.axis = None
        self.output_dtype = "int32"
        self.keepdim = True


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
        super().init_input_data()
        self.input_dtype = "float64"
        self.feed_data = {
            'x': self.random(self.shape, self.input_dtype),
        }


if __name__ == "__main__":
    unittest.main()