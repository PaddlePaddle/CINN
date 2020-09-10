#!/usr/bin/env python3
# import paddle as paddle
# import paddle.fluid as fluid
import numpy as np
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


class OpTest_relu(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X] = inputs_data
        return np.maximum(X, np.zeros(X.shape).astype("float32"))

    def test_op(self):
        attrs = framework.NodeAttr()
        self.to_test_op([[32]], [[32]], "relu", attrs)


class OpTest_relu6(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X] = inputs_data
        return np.minimum(
            np.maximum(X,
                       np.zeros(np.array(X).shape).astype("float32")), 6)

    def test_op(self):
        attrs = framework.NodeAttr()
        self.to_test_op([[32, 32]], [[32, 32]], "relu6", attrs)


""" class OpTest_conv2d(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        img = fluid.layers.data(name='img', shape=[1,3,10,10], dtype='float32')
        param = fluid.initializer.NumpyArrayInitializer(np.array(inputs_data[1]).reshape((2,3,2,2)).astype("float32"))
        res = fluid.layers.conv2d(input=img, num_filters=2, filter_size=2,stride = 2, padding = 1,dilation = 2, param_attr=param)
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        x = np.array(inputs_data[0]).reshape((1,3,10,10)).astype("float32")
        output = exe.run(feed={"img": x}, fetch_list=[res])
        output = np.array(output)
        print("output's shape is:", output.shape)
        return output

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.attr_store = {"padding": [1, 1]}
        attrs.set_attr("stride", [2, 2])
        attrs.set_attr("dilation", 2)
        attrs.set_attr("groups", 1)
        self.to_test_op([[1, 3, 10, 10], [2, 3, 2, 2]],
                        [[1, 3, 12, 12], [2, 3, 3, 3], [1, 2, 5, 5]], "conv2d", attrs) """


class OpTest_batchnorm(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X, Y] = inputs_data
        c = X.shape[1]
        for i in range(0, c):
            X[:, i, :, :] = (X[:, i, :, :] - Y[0, i]) / math.sqrt(
                Y[1, i] + 0.00001) * Y[2, i] + Y[3, i]
        return X

    def test_op(self):
        attrs = framework.NodeAttr()
        self.to_test_op([[1, 3, 2, 2], [4, 3]], [[1, 3, 2, 2]], "batchnorm",
                        attrs)


if __name__ == "__main__":
    unittest.main()
