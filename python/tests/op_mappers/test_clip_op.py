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

import paddle
import unittest
from op_mapper_test import OpMapperTest


class TestClipOp(OpMapperTest):

    def init_input_data(self):
        self.feed_data = {
            'x': self.random([2, 3, 4], "float32"),
        }

    def set_op_type(self):
        return "clip"

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype)
        return {'X': [x]}

    def set_op_attrs(self):
        return {"min": -0.2, "max": 0.2}

    def set_op_outputs(self):
        return {'Out': [str(self.feed_data['x'].dtype)]}

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestClipOpMaxTensor(TestClipOp):
    
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([2, 3, 4], "float32"),
            'max_input': self.random([1], "float32")
        }

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype)
        max_input = paddle.static.data(
            name='max_input',
            shape=self.feed_data['max_input'].shape,
            dtype=self.feed_data['max_input'].dtype)
        return {'X': [x], 'Max': [max_input]}

    def set_op_attrs(self):
        return {"min": 0.0, "max": 1.0}


class TestClipOpMinTensor(TestClipOp):
    
    def init_input_data(self):
        self.feed_data = {
            'x': self.random([2, 3, 4], "float32"),
            'min_input': self.random([1], "float32")
        }

    def set_op_inputs(self):
        x = paddle.static.data(
            name='x',
            shape=self.feed_data['x'].shape,
            dtype=self.feed_data['x'].dtype)
        min_input = paddle.static.data(
            name='min_input',
            shape=self.feed_data['min_input'].shape,
            dtype=self.feed_data['min_input'].dtype)
        return {'X': [x], 'Min': [min_input]}

    def set_op_attrs(self):
        return {"min": 0.0, "max": 1.0}


# class TestClipOpInt32(TestClipOp):

#     def init_input_data(self):
#         self.feed_data = {
#             'x': self.random([2, 3, 4], "int32", low=0, high=10),
#         }

#     def set_op_attrs(self):
#         return {"min": 3, "max": 7}

#     def set_op_outputs(self):
#         return {'Out': [str(self.feed_data['x'].dtype)]}

#     def test_check_results(self):
#         self.check_outputs_and_grads()


# class TestClipOpInt64(TestClipOpInt32):

#     def init_input_data(self):
#         self.feed_data = {
#             'x': self.random([2, 3, 4], "int64", low=0, high=10),
#         }


# class TestClipOpFloat64(TestClipOp):

#     def init_input_data(self):
#         self.feed_data = {
#             'x': self.random([2, 3, 4], "float64"),
#         }


# class TestClipOpNoMax(TestClipOp):
    
#     def set_op_attrs(self):
#         return {"min": -0.2}


# class TestClipOpNoMin(TestClipOp):
    
#     def set_op_attrs(self):
#         return {"max": 0.2}


# class TestClipOpNoMaxmin(TestClipOp):
    
#     def set_op_attrs(self):
#         return {}


if __name__ == "__main__":
    unittest.main()

