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


class OpTest_add_0(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X, Y] = inputs_data
        return X + Y

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.set_attr("axis", 0)
        self.to_test_op([[100, 32], [100, 32]], [[100, 32]], "elementwise_add",
                        attrs)


class OpTest_add_1(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X, Y] = inputs_data
        return X + Y

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.set_attr("axis", 1)
        self.to_test_op([[3, 2], [2]], [[3, 2]], "elementwise_add", attrs)


class OpTest_mul_0(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X, Y] = inputs_data
        return X * Y

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.set_attr("axis", 0)
        self.to_test_op([[100, 32], [100, 32]], [[100, 32]], "elementwise_mul",
                        attrs)


class OpTest_mul_1(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X, Y] = inputs_data
        return X * Y

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.set_attr("axis", 1)
        self.to_test_op([[3, 2], [2]], [[3, 2]], "elementwise_mul", attrs)


class OpTest_scale_0(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X] = inputs_data
        return X * attrs.attr_store["scale"] + attrs.attr_store["bias"]

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.set_attr("scale", 0.7)
        attrs.set_attr("bias", 0.3)
        self.to_test_op([[100, 32]], [[100, 32]], "scale", attrs)


class OpTest_scale_1(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X] = inputs_data
        return (X + attrs.attr_store["bias"]) * attrs.attr_store["scale"]

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.set_attr("scale", 0.6)
        attrs.set_attr("bias", 0.4)
        attrs.set_attr("bias_after_scale", False)
        self.to_test_op([[100, 32]], [[100, 32]], "scale", attrs)


if __name__ == "__main__":
    unittest.main()
