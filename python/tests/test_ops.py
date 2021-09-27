#!/usr/bin/env python3
import paddle
import paddle.fluid as fluid
from cinn.frontend import *
from cinn import Target
from cinn.framework import *
import unittest
import cinn
from cinn import runtime
from cinn import ir
from cinn import lang
from cinn.common import *
import numpy as np
import paddle.fluid as fluid
import sys

enable_gpu = sys.argv.pop()


class TestReshape(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

    def paddle_verify(self, result):
        paddle.enable_static()

        a = fluid.data(name='A', shape=[2, 2048, 1, 1], dtype='float32')
        res = fluid.layers.reshape(x=a, shape=[-1, 2048], inplace=True)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        x = np.array(result[0]).reshape((2, 2048, 1, 1)).astype("float32")
        output = exe.run(feed={"A": x}, fetch_list=[res])
        output = np.array(output).reshape(-1)
        print("result in paddle_verify: \n")
        for i in range(0, output.shape[0]):
            if np.abs(output[i] - result[len(result) - 1][i]) > 1e-4:
                print("Error! ", i, "-th data has diff with target data:\n",
                      output[i], " vs: ", result[len(result) - 1][i],
                      ". Diff is: ", output[i] - result[len(result) - 1][i])
        self.assertTrue(
            np.allclose(result[len(result) - 1], output, atol=1e-4))

    def test_basic(self):
        prog = Program()

        a = Variable("A").set_type(Float(32)).set_shape([2, 2048, 1, 1])
        h = prog.reshape(a, {-1, 2048})
        self.assertEqual(prog.size(), 1)
        # print program
        for i in range(prog.size()):
            print(prog[i])
        tensor_data = [np.random.random([2, 2048, 1, 1]).astype("float32")]
        result = prog.build_and_get_output(self.target, [a], tensor_data, h)
        result = result.numpy(self.target).reshape(-1)
        tensor_data.append(result)
        self.paddle_verify(tensor_data)


class TestConcat(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

    def paddle_verify(self, result):
        paddle.enable_static()

        a = fluid.data(name='A', shape=[1, 256, 14, 14], dtype='float32')
        b = fluid.data(name='B', shape=[1, 256, 14, 14], dtype='float32')
        res = fluid.layers.concat(input=[a, b], axis=1)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        x = np.array(result[0]).reshape((1, 256, 14, 14)).astype("float32")
        y = np.array(result[0]).reshape((1, 256, 14, 14)).astype("float32")
        output = exe.run(feed={"A": x, "B": y}, fetch_list=[res])
        output = np.array(output).reshape(-1)
        print("result in paddle_verify: \n")
        for i in range(0, output.shape[0]):
            if np.abs(output[i] - result[len(result) - 1][i]) > 1e-4:
                print("Error! ", i, "-th data has diff with target data:\n",
                      output[i], " vs: ", result[len(result) - 1][i],
                      ". Diff is: ", output[i] - result[len(result) - 1][i])
        self.assertTrue(
            np.allclose(result[len(result) - 1], output, atol=1e-4))

    def test_basic(self):
        prog = Program()

        a = Variable("A").set_type(Float(32)).set_shape([1, 256, 14, 14])
        b = Variable("B").set_type(Float(32)).set_shape([1, 256, 14, 14])
        h = prog.concat(a, b, 1)
        self.assertEqual(prog.size(), 1)
        # print program
        for i in range(prog.size()):
            print(prog[i])
        tensor_data = [
            np.random.random([1, 256, 14, 14]).astype("float32"),
            np.random.random([1, 256, 14, 14]).astype("float32")
        ]
        result = prog.build_and_get_output(self.target, [a, b], tensor_data, h)
        result = result.numpy(self.target).reshape(-1)
        tensor_data.append(result)
        self.paddle_verify(tensor_data)
