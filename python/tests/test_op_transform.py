#!/usr/bin/env python3
import unittest
import math
import numpy as np
import cinn
from cinn import frontend
from cinn import runtime
from cinn import lang
from cinn import framework
from cinn import ir
from cinn import common
from cinn.poly import create_stages
import logging
from test_utils import SingleOpTester
import paddle
import paddle.fluid as fluid


def matmul_util(inputs_data, input_shape, trans_a, trans_b, alpha):
    main_program = fluid.Program()
    paddle.enable_static()
    with fluid.program_guard(main_program, fluid.Program()):
        [input_x, input_y] = inputs_data
        x = fluid.layers.data(name='x', shape=input_shape[0], dtype='float32')
        y = fluid.layers.data(name='y', shape=input_shape[1], dtype='float32')
        output = fluid.layers.matmul(x, y, trans_a, trans_b, alpha)
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())
        res, = exe.run(
            fluid.default_main_program(),
            feed={
                'x': input_x,
                'y': input_y
            },
            fetch_list=[output])
        return res


class OpTest_matmul_0(SingleOpTester):
    def init_testcase(self):
        self.input_shape = [[100, 32], [32, 100]]
        self.output_shape = [[100, 100], [100, 100]]
        self.trans_a = False
        self.trans_b = False
        self.alpha = 1
        self.attrs = framework.NodeAttr()
        self.attrs.attr_store = {
            "trans_a": self.trans_a,
            "trans_b": self.trans_b,
            "alpha": self.alpha
        }

    def create_target_data(self, inputs_data, attrs):
        return matmul_util(inputs_data, self.input_shape, self.trans_a,
                           self.trans_b, self.alpha)

    def test_op(self):
        self.init_testcase()
        self.to_test_op(self.input_shape, self.output_shape, "matmul",
                        self.attrs, 0)


class OpTest_matmul_1(SingleOpTester):
    def init_testcase(self):
        # self.input_shape = [[256, 32], [256, 32]]
        # self.output_shape = [[256, 256], [256, 256], [4, 32, 64]]
        self.input_shape = [[100, 32], [100, 32]]
        self.output_shape = [[100, 100], [100, 100], [2, 32, 50]]
        self.trans_a = False
        self.trans_b = True
        self.alpha = 2
        self.attrs = framework.NodeAttr()
        self.attrs.attr_store = {
            "trans_a": self.trans_a,
            "trans_b": self.trans_b,
            "alpha": self.alpha
        }

    def create_target_data(self, inputs_data, attrs):
        return matmul_util(inputs_data, self.input_shape, self.trans_a,
                           self.trans_b, self.alpha)

    def test_op(self):
        self.init_testcase()
        self.to_test_op(self.input_shape, self.output_shape, "matmul",
                        self.attrs, 0)


class OpTest_matmul_2(SingleOpTester):
    def init_testcase(self):
        self.input_shape = [[2, 3, 100, 32], [2, 3, 100, 32]]
        self.output_shape = [[2, 3, 100, 100], [2, 3, 100, 100],
                             [2, 3, 2, 100, 16]]
        self.trans_a = False
        self.trans_b = True
        self.alpha = 2
        self.attrs = framework.NodeAttr()
        self.attrs.attr_store = {
            "trans_a": self.trans_a,
            "trans_b": self.trans_b,
            "alpha": self.alpha
        }

    def create_target_data(self, inputs_data, attrs):
        return matmul_util(inputs_data, self.input_shape, self.trans_a,
                           self.trans_b, self.alpha)

    def test_op(self):
        self.init_testcase()
        self.to_test_op(self.input_shape, self.output_shape, "matmul",
                        self.attrs, 0)


class OpTest_matmul_3(SingleOpTester):
    def init_testcase(self):
        self.input_shape = [[32, 100], [32, 100]]
        self.output_shape = [[100, 100], [100, 100], [2, 100, 16]]
        self.trans_a = True
        self.trans_b = False
        self.alpha = 2
        self.attrs = framework.NodeAttr()
        self.attrs.attr_store = {
            "trans_a": self.trans_a,
            "trans_b": self.trans_b,
            "alpha": self.alpha
        }

    def create_target_data(self, inputs_data, attrs):
        return matmul_util(inputs_data, self.input_shape, self.trans_a,
                           self.trans_b, self.alpha)

    def test_op(self):
        self.init_testcase()
        self.to_test_op(self.input_shape, self.output_shape, "matmul",
                        self.attrs, 0)


class OpTest_matmul_4(SingleOpTester):
    def init_testcase(self):
        self.input_shape = [[32, 100], [100]]
        self.output_shape = [[32], [32], [2, 100, 16]]
        self.trans_a = False
        self.trans_b = False
        self.alpha = 2
        self.attrs = framework.NodeAttr()
        self.attrs.attr_store = {
            "trans_a": self.trans_a,
            "trans_b": self.trans_b,
            "alpha": self.alpha
        }

    def create_target_data(self, inputs_data, attrs):
        return matmul_util(inputs_data, self.input_shape, self.trans_a,
                           self.trans_b, self.alpha)

    def test_op(self):
        self.init_testcase()
        self.to_test_op(self.input_shape, self.output_shape, "matmul",
                        self.attrs, 0)


class OpTest_matmul_5(SingleOpTester):
    def init_testcase(self):
        self.input_shape = [[100], [100]]
        self.output_shape = [[1], [1], [1, 100, 1]]
        self.trans_a = False
        self.trans_b = False
        self.alpha = 2
        self.attrs = framework.NodeAttr()
        self.attrs.attr_store = {
            "trans_a": self.trans_a,
            "trans_b": self.trans_b,
            "alpha": self.alpha
        }

    def create_target_data(self, inputs_data, attrs):
        return matmul_util(inputs_data, self.input_shape, self.trans_a,
                           self.trans_b, self.alpha)

    def test_op(self):
        self.init_testcase()
        self.to_test_op(self.input_shape, self.output_shape, "matmul",
                        self.attrs, 0)


class OpTest_matmul_6(SingleOpTester):
    def init_testcase(self):
        self.input_shape = [[32, 1], [1, 100]]
        self.output_shape = [[32, 100], [32, 100], [2, 1, 50]]
        self.trans_a = False
        self.trans_b = False
        self.alpha = 2
        self.attrs = framework.NodeAttr()
        self.attrs.attr_store = {
            "trans_a": self.trans_a,
            "trans_b": self.trans_b,
            "alpha": self.alpha
        }

    def create_target_data(self, inputs_data, attrs):
        return matmul_util(inputs_data, self.input_shape, self.trans_a,
                           self.trans_b, self.alpha)

    def test_op(self):
        self.init_testcase()
        self.to_test_op(self.input_shape, self.output_shape, "matmul",
                        self.attrs, 0)


if __name__ == "__main__":
    unittest.main()
