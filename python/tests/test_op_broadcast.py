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
    def create_target_data(self, inputs_data):
        [X, Y] = inputs_data
        return X + Y

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.attr_store = {"axis": 0}
        self.to_test_op([[100, 32], [100, 32]], [[100, 32]], "elementwise_add",
                        attrs)


class OpTest_add_1(SingleOpTester):
    def create_target_data(self, inputs_data):
        [X, Y] = inputs_data
        return X + Y

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.attr_store = {"axis": 1}
        self.to_test_op([[3, 2], [2]], [[3, 2]], "elementwise_add", attrs)


if __name__ == "__main__":
    unittest.main()
