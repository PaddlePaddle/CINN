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

import unittest
import numpy as np
from op_test import OpTest, OpTestTool
import paddle
import cinn
from cinn.frontend import *
from cinn.common import *


@OpTestTool.skip_if(not is_compiled_with_cuda(),
                    "x86 test will be skipped due to timeout.")
class TestReduceBaseOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", -1.0, 1.0)}
        self.dim = []
        self.keep_dim = False

    def paddle_func(self, x):
        return paddle.sum(x, axis=self.dim, keepdim=self.keep_dim)

    def cinn_func(self, builder, x):
        return builder.reduce_sum(x, self.dim, self.keep_dim)

    def cinn_create_input(self, builder, shape, name):
        return builder.create_input(Float(32), shape, name)

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        out = self.paddle_func(x)
        self.paddle_outputs = [out]

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("reduce")
        x = self.cinn_create_input(builder, self.inputs["x"].shape, "x")
        out = self.cinn_func(builder, x)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]],
                                   [out])

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestReduceSumOp(TestReduceBaseOp):
    def paddle_func(self, x):
        return paddle.sum(x, axis=self.dim, keepdim=self.keep_dim)

    def cinn_func(self, builder, x):
        return builder.reduce_sum(x, self.dim, self.keep_dim)


class TestReduceSumCase1(TestReduceSumOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", -1.0, 1.0)}
        self.dim = [1]
        self.keep_dim = False


class TestReduceSumCase2(TestReduceSumOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", -1.0, 1.0)}
        self.dim = [0, 1]
        self.keep_dim = False


class TestReduceSumCase3(TestReduceSumOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", -1.0, 1.0)}
        self.dim = [0, 2]
        self.keep_dim = False


class TestReduceSumCase4(TestReduceSumOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", -1.0, 1.0)}
        self.dim = [0]
        self.keep_dim = True


class TestReduceSumCase5(TestReduceSumOp):
    def init_case(self):
        self.inputs = {"x": self.random([1, 16, 256], "float32", -1.0, 1.0)}
        self.dim = [0]
        self.keep_dim = False


class TestReduceSumCase6(TestReduceSumOp):
    def init_case(self):
        self.inputs = {"x": self.random([1, 12, 9, 9], "float32", -1.0, 1.0)}
        self.dim = [-1]
        self.keep_dim = False


class TestReduceSumCase7(TestReduceSumOp):
    def init_case(self):
        self.inputs = {
            "x": self.random([1, 1, 10, 10, 10], "float32", -1.0, 1.0)
        }
        self.dim = [0]
        self.keep_dim = False


class TestReduceSumCase8(TestReduceSumOp):
    def init_case(self):
        self.inputs = {
            "x": self.random([1, 10, 1, 10, 10], "float32", -1.0, 1.0)
        }
        self.dim = [0, 2]
        self.keep_dim = False


class TestReduceSumCase9(TestReduceSumOp):
    def init_case(self):
        self.inputs = {"x": self.random([1, 1, 10], "float32", -1.0, 1.0)}
        self.dim = [0, 2]
        self.keep_dim = False


class TestReduceSumCase10(TestReduceSumOp):
    def init_case(self):
        self.inputs = {"x": self.random([1, 1, 10], "float32", -1.0, 1.0)}
        self.dim = [0, 2]
        self.keep_dim = True


class TestReduceSumCase11(TestReduceSumOp):
    def init_case(self):
        self.inputs = {
            "x": self.random([32, 32, 32, 32], "float32", -1.0, 1.0)
        }
        self.dim = [0, 2, 3]
        self.keep_dim = False

    def test_check_results(self):
        # the shape of tensor is large, lead to the different of result increase
        self.check_outputs_and_grads(max_relative_error=1e-4)


class TestReduceSumCase12(TestReduceSumOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 1024], "float32", -1.0, 1.0)}
        self.dim = []
        self.keep_dim = False


class TestReduceSumCase13(TestReduceSumOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 1024], "float32", -1.0, 1.0)}
        self.dim = [0]
        self.keep_dim = False


class TestReduceSumCase14(TestReduceSumOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 1024], "float32", -1.0, 1.0)}
        self.dim = [1]
        self.keep_dim = False


class TestReduceSumCase15(TestReduceSumOp):
    def init_case(self):
        # data shape from resnet50 bs=64
        self.inputs = {
            "x": self.random([128, 64, 56, 56], "float32", -1.0, 1.0)
        }
        self.dim = [0, 2, 3]
        self.keep_dim = False

    def test_check_results(self):
        # the shape of tensor is large, lead to the different of result increase
        self.check_outputs_and_grads(max_relative_error=1e-3)


class TestReduceSumCase16(TestReduceSumOp):
    def init_case(self):
        # data shape from resnet50 NHWC bs=64
        self.inputs = {
            "x": self.random([128, 56, 56, 64], "float32", -1.0, 1.0)
        }
        self.dim = [0, 1, 2]
        self.keep_dim = False

    def test_check_results(self):
        # the shape of tensor is large, lead to the different of result increase
        self.check_outputs_and_grads(max_relative_error=1e-3)


'''
class TestReduceSumFP64(TestReduceSumOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float64", -1.0, 1.0)}
        self.dim = []
        self.keep_dim = False

    def cinn_create_input(self, builder, shape, name):
        return builder.create_input(Float(64), shape, name)


class TestReduceProdOp(TestReduceBaseOp):
    def paddle_func(self, x):
        return paddle.prod(x, axis=self.dim, keepdim=self.keep_dim)

    def cinn_func(self, builder, x):
        return builder.reduce_prod(x, self.dim, self.keep_dim)


class TestReduceProdCase1(TestReduceProdOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", -1.0, 1.0)}
        self.dim = [0, 1]
        self.keep_dim = False


class TestReduceProdCase2(TestReduceProdOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", -1.0, 1.0)}
        self.dim = [0, 1]
        self.keep_dim = True


class TestReduceProdCase3(TestReduceProdOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", -1.0, 1.0)}
        self.dim = [0]
        self.keep_dim = False


class TestReduceProdCase4(TestReduceProdOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", -1.0, 1.0)}
        self.dim = []
        self.keep_dim = False


class TestReduceProdFP64(TestReduceProdOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float64", -1.0, 1.0)}
        self.dim = []
        self.keep_dim = False

    def cinn_create_input(self, builder, shape, name):
        return builder.create_input(Float(64), shape, name)


class TestReduceMaxOp(TestReduceBaseOp):
    def paddle_func(self, x):
        return paddle.max(x, axis=self.dim, keepdim=self.keep_dim)

    def cinn_func(self, builder, x):
        return builder.reduce_max(x, self.dim, self.keep_dim)

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestReduceMaxCase1(TestReduceMaxOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", -1.0, 1.0)}
        self.dim = [0, 1]
        self.keep_dim = False


class TestReduceMaxCase2(TestReduceMaxOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", -1.0, 1.0)}
        self.dim = [0, 1]
        self.keep_dim = True


class TestReduceMaxCase3(TestReduceMaxOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", -1.0, 1.0)}
        self.dim = [0]
        self.keep_dim = False


class TestReduceMaxCase4(TestReduceMaxOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", -10.0, -1.0)}
        self.dim = []
        self.keep_dim = False


class TestReduceMaxFP64(TestReduceMaxOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float64", -1.0, 1.0)}
        self.dim = []
        self.keep_dim = False

    def cinn_create_input(self, builder, shape, name):
        return builder.create_input(Float(64), shape, name)


class TestReduceMaxFP64Case1(TestReduceMaxFP64):
    def init_case(self):
        self.inputs = {"x": self.random([2, 3, 4, 5], "float64", -1.0, 1.0)}
        self.dim = [0, 1]
        self.keep_dim = False


class TestReduceMinOp(TestReduceBaseOp):
    def paddle_func(self, x):
        return paddle.min(x, axis=self.dim, keepdim=self.keep_dim)

    def cinn_func(self, builder, x):
        return builder.reduce_min(x, self.dim, self.keep_dim)

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestReduceMinCase1(TestReduceMinOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", -1.0, 1.0)}
        self.dim = [0, 1]
        self.keep_dim = False


class TestReduceMinCase2(TestReduceMinOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", -1.0, 1.0)}
        self.dim = [0, 1]
        self.keep_dim = True


class TestReduceMinCase3(TestReduceMinOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", -1.0, 1.0)}
        self.dim = [0]
        self.keep_dim = False


class TestReduceMinCase4(TestReduceMinOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", 1.0, 10.0)}
        self.dim = []
        self.keep_dim = False


class TestReduceMinFP64(TestReduceMinOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float64", -1.0, 1.0)}
        self.dim = []
        self.keep_dim = False

    def cinn_create_input(self, builder, shape, name):
        return builder.create_input(Float(64), shape, name)


class TestReduceMinFP64Case1(TestReduceMinFP64):
    def init_case(self):
        self.inputs = {"x": self.random([2, 3, 4, 5], "float64", -1.0, 1.0)}
        self.dim = [0, 1]
        self.keep_dim = False


class TestAllOp(TestReduceBaseOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "bool")}
        self.dim = []
        self.keep_dim = False

    def paddle_func(self, x):
        return paddle.all(x, axis=self.dim, keepdim=self.keep_dim)

    def cinn_func(self, builder, x):
        return builder.reduce_all(x, self.dim, self.keep_dim)

    def cinn_create_input(self, builder, shape, name):
        return builder.create_input(Bool(), shape, name)

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestAllCase1(TestAllOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "bool")}
        self.dim = []
        self.keep_dim = True


class TestAllCase2(TestAllOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "bool")}
        self.dim = [0, 1]
        self.keep_dim = False


class TestAllCase3(TestAllOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "bool")}
        self.dim = [0, 1]
        self.keep_dim = True


class TestAllCase4(TestAllOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "bool")}
        self.dim = [0]
        self.keep_dim = False


class TestAllCase5(TestAllOp):
    def init_case(self):
        self.inputs = {"x": np.full([10, 10, 10], True, 'bool')}
        self.dim = []
        self.keep_dim = False


class TestAnyOp(TestReduceBaseOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "bool")}
        self.dim = []
        self.keep_dim = False

    def paddle_func(self, x):
        return paddle.any(x, axis=self.dim, keepdim=self.keep_dim)

    def cinn_func(self, builder, x):
        return builder.reduce_any(x, self.dim, self.keep_dim)

    def cinn_create_input(self, builder, shape, name):
        return builder.create_input(Bool(), shape, name)

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestAnyCase1(TestAnyOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "bool")}
        self.dim = []
        self.keep_dim = True


class TestAnyCase2(TestAnyOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "bool")}
        self.dim = [0, 1]
        self.keep_dim = False


class TestAnyCase3(TestAnyOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "bool")}
        self.dim = [0, 1]
        self.keep_dim = True


class TestAnyCase4(TestAnyOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "bool")}
        self.dim = [0]
        self.keep_dim = False


class TestAnyCase5(TestAllOp):
    def init_case(self):
        self.inputs = {"x": np.full([10, 10, 10], False, 'bool')}
        self.dim = []
        self.keep_dim = False
'''

if __name__ == "__main__":
    unittest.main()
