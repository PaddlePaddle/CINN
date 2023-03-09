#!/usr/bin/env python3

# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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
import numpy as np
from op_mapper_test import OpMapperTest, logger
import paddle


class TestSplitOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "float32"),
        }
        self.axis = 0
        self.num = 4

    def set_op_type(self):
        return "split"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype)
        return {'X': [x]}

    def set_op_attrs(self):
        return {"axis": 0, "num": self.num}

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['x'].dtype)] * self.num}

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestSplitCase1(TestSplitOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "float32"),
        }
        self.axis = -1
        self.num = 8


class TestSplitWithSection(TestSplitOp):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "float32"),
        }
        self.axis = 0
        self.sections = [3, 5, 16, 2, 6]

    def set_op_attrs(self):
        return {"axis": 0, "sections": self.sections}

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['x'].dtype)] * len(self.sections)}


class TestSplitWithSectionCase1(TestSplitWithSection):
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([32, 64], "int32", 0, 10000),
        }
        self.axis = 0
        self.sections = [4, 4, 16, 8]


if __name__ == "__main__":
    unittest.main()
