#!/usr/bin/env python3
import paddle as paddle
import paddle.fluid as fluid
import numpy as np
import unittest
import math
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
import pool_utils


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


class OpTest_conv2d(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        img = fluid.layers.data(name='img', shape=[3, 10, 10], dtype='float32')
        param = fluid.initializer.NumpyArrayInitializer(
            np.array(inputs_data[1]).reshape((2, 3, 2, 2)).astype("float32"))
        res = fluid.layers.conv2d(
            input=img,
            num_filters=2,
            filter_size=2,
            stride=2,
            padding=1,
            dilation=2,
            param_attr=param)
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        x = np.array(inputs_data[0]).reshape((1, 3, 10, 10)).astype("float32")
        output = exe.run(feed={"img": x}, fetch_list=[res])
        output = np.array(output)
        print("output's shape is:", output.shape)
        return output

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.attr_store = {"padding": [1, 1]}
        attrs.set_attr("stride", [2, 2])
        attrs.set_attr("dilation", [2, 2])
        attrs.set_attr("groups", 1)
        self.to_test_op([[1, 3, 10, 10], [2, 3, 2, 2]],
                        [[1, 3, 12, 12], [2, 3, 3, 3], [1, 2, 5, 5]], "conv2d",
                        attrs)


class OpTest_pool1d(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.attr_store = {
        "kernel_size": [2],
        "stride_size": [2],
        "padding_size": [1, 1],
        "pool_type": "max",
        "ceil_mode": False,
        "exclusive": True,
        "data_format": "NCW"
    }

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool1d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 3, 8]
        self.to_test_op([input_shape], None, "pool1d", self.attrs)


class OpTest_pool1d_1(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.attr_store = {
        "kernel_size": [2],
        "stride_size": [2],
        "padding_size": [2, 3],
        "pool_type": "avg",
        "ceil_mode": False,
        "exclusive": True,
        "data_format": "NCW"
    }

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool1d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 3, 8]
        self.to_test_op([input_shape], None, "pool1d", self.attrs)


class OpTest_pool1d_2(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.attr_store = {
        "kernel_size": [2],
        "stride_size": [3],
        "padding_size": [4, 5],
        "pool_type": "avg",
        "ceil_mode": True,
        "exclusive": False,
        "data_format": "NWC"
    }

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool1d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 8, 3]
        self.to_test_op([input_shape], None, "pool1d", self.attrs)


class OpTest_pool2d(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.attr_store = {
        "kernel_size": [2, 2],
        "stride_size": [2, 2],
        "padding_size": [1, 1, 1, 1],
        "pool_type": "max",
        "ceil_mode": False,
        "exclusive": True,
        "data_format": "NCHW"
    }

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool2d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 3, 8, 8]
        self.to_test_op([input_shape], None, "pool2d", self.attrs)


class OpTest_pool2d_1(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.attr_store = {
        "kernel_size": [2, 2],
        "stride_size": [2, 2],
        "padding_size": [2, 3, 4, 5],
        "pool_type": "avg",
        "ceil_mode": False,
        "exclusive": True,
        "data_format": "NCHW"
    }

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool2d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 3, 8, 8]
        self.to_test_op([input_shape], None, "pool2d", self.attrs)


class OpTest_pool2d_2(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.attr_store = {
        "kernel_size": [2, 2],
        "stride_size": [3, 3],
        "padding_size": [2, 3, 4, 5],
        "pool_type": "avg",
        "ceil_mode": True,
        "exclusive": False,
        "data_format": "NHWC"
    }

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool2d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 8, 8, 3]
        self.to_test_op([input_shape], None, "pool2d", self.attrs)


class OpTest_pool3d(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.attr_store = {
        "kernel_size": [2, 2, 2],
        "stride_size": [2, 2, 2],
        "padding_size": [1, 2, 3, 4, 5, 6],
        "pool_type": "max",
        "ceil_mode": False,
        "exclusive": True,
        "data_format": "NCDHW"
    }

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool3d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [2, 3, 8, 8, 8]
        self.to_test_op([input_shape], None, "pool3d", self.attrs)


class OpTest_pool3d_1(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.attr_store = {
        "kernel_size": [2, 2, 2],
        "stride_size": [2, 2, 2],
        "padding_size": [1, 1, 1, 1, 1, 1],
        "pool_type": "avg",
        "ceil_mode": False,
        "exclusive": True,
        "data_format": "NCDHW"
    }

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool3d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 3, 8, 8, 8]
        self.to_test_op([input_shape], None, "pool3d", self.attrs)


class OpTest_pool3d_2(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.attr_store = {
        "kernel_size": [2, 2, 2],
        "stride_size": [2, 2, 2],
        "padding_size": [1, 2, 3, 4, 5, 6],
        "pool_type": "avg",
        "ceil_mode": True,
        "exclusive": False,
        "data_format": "NDHWC"
    }

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool3d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 8, 8, 8, 3]
        self.to_test_op([input_shape], None, "pool3d", self.attrs)


class OpTest_pool1d(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.attr_store = {
        "kernel_size": [2],
        "stride_size": [2],
        "padding_size": [1, 1],
        "pool_type": "max",
        "ceil_mode": False,
        "exclusive": True,
        "data_format": "NCW"
    }

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool1d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 3, 8]
        self.to_test_op([input_shape], None, "pool1d", self.attrs)


class OpTest_pool1d_1(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.attr_store = {
        "kernel_size": [2],
        "stride_size": [2],
        "padding_size": [2, 3],
        "pool_type": "avg",
        "ceil_mode": False,
        "exclusive": True,
        "data_format": "NCW"
    }

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool1d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 3, 8]
        self.to_test_op([input_shape], None, "pool1d", self.attrs)


class OpTest_pool1d_2(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.attr_store = {
        "kernel_size": [2],
        "stride_size": [3],
        "padding_size": [4, 5],
        "pool_type": "avg",
        "ceil_mode": True,
        "exclusive": False,
        "data_format": "NWC"
    }

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool1d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 8, 3]
        self.to_test_op([input_shape], None, "pool1d", self.attrs)


class OpTest_pool2d(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.attr_store = {
        "kernel_size": [2, 2],
        "stride_size": [2, 2],
        "padding_size": [1, 1, 1, 1],
        "pool_type": "max",
        "ceil_mode": False,
        "exclusive": True,
        "data_format": "NCHW"
    }

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool2d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 3, 8, 8]
        self.to_test_op([input_shape], None, "pool2d", self.attrs)


class OpTest_pool2d_1(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.attr_store = {
        "kernel_size": [2, 2],
        "stride_size": [2, 2],
        "padding_size": [2, 3, 4, 5],
        "pool_type": "avg",
        "ceil_mode": False,
        "exclusive": True,
        "data_format": "NCHW"
    }

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool2d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 3, 8, 8]
        self.to_test_op([input_shape], None, "pool2d", self.attrs)


class OpTest_pool2d_2(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.attr_store = {
        "kernel_size": [2, 2],
        "stride_size": [3, 3],
        "padding_size": [2, 3, 4, 5],
        "pool_type": "avg",
        "ceil_mode": True,
        "exclusive": False,
        "data_format": "NHWC"
    }

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool2d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 8, 8, 3]
        self.to_test_op([input_shape], None, "pool2d", self.attrs)


class OpTest_pool3d(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.attr_store = {
        "kernel_size": [2, 2, 2],
        "stride_size": [2, 2, 2],
        "padding_size": [1, 2, 3, 4, 5, 6],
        "pool_type": "max",
        "ceil_mode": False,
        "exclusive": True,
        "data_format": "NCDHW"
    }

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool3d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [2, 3, 8, 8, 8]
        self.to_test_op([input_shape], None, "pool3d", self.attrs)


class OpTest_pool3d_1(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.attr_store = {
        "kernel_size": [2, 2, 2],
        "stride_size": [2, 2, 2],
        "padding_size": [1, 1, 1, 1, 1, 1],
        "pool_type": "avg",
        "ceil_mode": False,
        "exclusive": True,
        "data_format": "NCDHW"
    }

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool3d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 3, 8, 8, 8]
        self.to_test_op([input_shape], None, "pool3d", self.attrs)


class OpTest_pool3d_2(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.attr_store = {
        "kernel_size": [2, 2, 2],
        "stride_size": [2, 2, 2],
        "padding_size": [1, 2, 3, 4, 5, 6],
        "pool_type": "avg",
        "ceil_mode": True,
        "exclusive": False,
        "data_format": "NDHWC"
    }

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool3d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 8, 8, 8, 3]
        self.to_test_op([input_shape], None, "pool3d", self.attrs)


class OpTest_batchnorm(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X, Scale, Bias, Mean, Variance] = inputs_data
        c = X.shape[1]
        for i in range(0, c):
            """ TODO(haozech) This should be the correct compute function(with sqrt)
            X[:, i, :, :] = (X[:, i, :, :] - Mean[i]) / math.sqrt(
                Variance[i] + 0.00001) * Scale[i] + Bias[i] """
            X[:, i, :, :] = (X[:, i, :, :] - Mean[i]) / (
                Variance[i] + 0.00001) * Scale[i] + Bias[i]
        return X

    def test_op(self):
        attrs = framework.NodeAttr()
        self.to_test_op([[1, 64, 112, 112], [64], [64], [64], [64]],
                        [[1, 64, 112, 112]], "batchnorm", attrs)


class OpTest_softmax_0(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X] = inputs_data
        Y = np.zeros(X.shape).astype("float32")
        for i in range(0, Y.shape[1]):
            Y[:, i, :] = np.exp(X[:, i, :]) / np.sum(
                np.exp(X), axis=1, keepdims=True)[:, 0, :]
        return Y

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.set_attr("axis", 1)
        self.to_test_op([[12, 224, 224]], [[12, 224, 224], [12, 224, 224]],
                        "softmax", attrs)


class OpTest_softmax_1(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X] = inputs_data
        Y = np.zeros(X.shape).astype("float32")
        for i in range(0, Y.shape[2]):
            Y[:, :, i] = np.exp(X[:, :, i]) / np.sum(
                np.exp(X), axis=2, keepdims=True)[:, :, 0]
        return Y

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.set_attr("axis", -1)
        self.to_test_op([[12, 224, 224]], [[12, 224, 224], [12, 224, 224]],
                        "softmax", attrs)


class OpTest_softmax_2(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X] = inputs_data
        Y = np.zeros(X.shape).astype("float32")
        for i in range(0, Y.shape[0]):
            Y[i, :, :] = np.exp(X[i, :, :]) / np.sum(
                np.exp(X), axis=0, keepdims=True)[0, :, :]
        return Y

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.set_attr("axis", 0)
        self.to_test_op([[12, 224, 224]], [[12, 224, 224], [12, 224, 224]],
                        "softmax", attrs)


class OpTest_sigmoid(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        x = np.array(inputs_data[0])
        y = 1 / (1 + np.exp(-x))
        print("output's shape is:", y.shape)
        return y

    def test_op(self):
        attrs = framework.NodeAttr()
        self.to_test_op([[3, 224, 224]], [[3, 224, 224]], "sigmoid", attrs)


if __name__ == "__main__":
    unittest.main()
