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

import unittest
import numpy as np
from op_mapper_test import OpMapperTest, logger
import paddle


class TestRandIntOp(OpMapperTest):
    def init_input_data(self):
        self.feed_data = dict()
        self.shape = [2, 3]
        self.min = 1
        self.max = 5
        self.seed = 10
        self.dtype = "int32"

    def set_op_type(self):
        return "randint"

    def set_op_inputs(self):
        return {}

    def set_op_attrs(self):
        return {
            "low": self.min,
            "high": self.max,
            "seed": self.seed,
            "shape": self.shape,
            "dtype": self.nptype2paddledtype(self.dtype)
        }

    def set_op_outputs(self):
        return {'Out': [self.dtype]}

    def test_check_results(self):
        # Due to the different random number generation numbers implemented
        # in the specific implementation, the random number results generated
        # by CINN and Paddle are not the same, but they all conform to the
        # uniform distribution.
        self.build_paddle_program(self.target)


class TestRandIntCase1(TestRandIntOp):
    def init_input_data(self):
        self.feed_data = dict()
        self.shape = [2, 3, 4]
        self.min = 1
        self.max = 9
        self.seed = 10
        self.dtype = "int32"


class TestRandIntCase2(TestRandIntOp):
    def init_input_data(self):
        self.feed_data = dict()
        self.shape = [2, 3, 4]
        self.min = 1
        self.max = 9
        self.seed = 10
        self.dtype = "int64"


if __name__ == "__main__":
    unittest.main()
