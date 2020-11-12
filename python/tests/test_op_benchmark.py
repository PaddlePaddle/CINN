#!/usr/bin/env python3
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
import sys

assert len(sys.argv) == 2
enable_gpu = sys.argv.pop()


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

    def test_conv2d(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([2, 512, 7, 7])
        b = Variable("E").set_type(Float(32)).set_shape([512, 512, 3, 3])
        c = prog.conv2d(a, b, {
            "stride": [1, 1],
            "dilation": [1, 1],
            "padding": [1, 1]
        })
        tensor_data = [
            np.random.random([2, 512, 7, 7]).astype("float32"),
            np.random.random([512, 512, 3, 3]).astype("float32")
        ]
        result = prog.test_benchmark(
            self.target, [a, b], tensor_data, c, 200,
            "TESTING [conv2d] time cost with shape [2,512,7,7]...")

    def test_softmax(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([1024, 2048])
        c = prog.softmax(a, {})
        tensor_data = [np.random.random([1024, 2048]).astype("float32")]
        result = prog.test_benchmark(
            self.target, [a], tensor_data, c, 200,
            "TESTING [softmax] time cost with shape [1024,2048]...")

    def test_matmul(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([512, 512])
        b = Variable("B").set_type(Float(32)).set_shape([512, 512])
        c = prog.mul(a, b, 1, 1)
        tensor_data = [
            np.random.random([512, 512]).astype("float32"),
            np.random.random([512, 512]).astype("float32")
        ]
        result = prog.test_benchmark(
            self.target, [a, b], tensor_data, c, 200,
            "TESTING [matmul] time cost with shape [512,512]...")

    def test_pool2d(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([2, 64, 112, 112])
        c = prog.pool2d(
            a, {
                "kernel_size": (3, 3),
                "stride_size": (2, 2),
                "padding_size": (1, 1, 1, 1)
            })
        tensor_data = [np.random.random([2, 64, 112, 112]).astype("float32")]
        result = prog.test_benchmark(
            self.target, [a], tensor_data, c, 200,
            "TESTING [pool2d] time cost with shape [2, 64, 112, 112]...")

    def test_elementwise(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([64, 64])
        b = Variable("B").set_type(Float(32)).set_shape([64, 64])
        c = prog.add(a, b)
        tensor_data = [
            np.random.random([64, 64]).astype("float32"),
            np.random.random([64, 64]).astype("float32")
        ]
        result = prog.test_benchmark(
            self.target, [a, b], tensor_data, c, 200,
            "TESTING [elementwise_add] time cost with shape [64,64]...")

    def test_batchnorm(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([2, 512, 7, 7])
        b = Variable("B").set_type(Float(32)).set_shape([512])
        c = Variable("C").set_type(Float(32)).set_shape([512])
        d = Variable("D").set_type(Float(32)).set_shape([512])
        e = Variable("E").set_type(Float(32)).set_shape([512])
        f = prog.batchnorm(a, b, c, d, e, {})
        tensor_data = [
            np.random.random([2, 512, 7, 7]).astype("float32"),
            np.random.random([512]).astype("float32"),
            np.random.random([512]).astype("float32"),
            np.random.random([512]).astype("float32"),
            np.random.random([512]).astype("float32")
        ]
        result = prog.test_benchmark(
            self.target, [a, b, c, d, e], tensor_data, f, 200,
            "TESTING [batchnorm] time cost with shape [2, 512, 7, 7]...")

    def test_relu(self):
        prog = Program()
        a = Variable("A").set_type(Float(32)).set_shape([64, 64])
        c = prog.relu(a)
        tensor_data = [np.random.random([64, 64]).astype("float32")]
        result = prog.test_benchmark(
            self.target, [a], tensor_data, c, 200,
            "TESTING [relu] time cost with shape [64,64]...")


if __name__ == "__main__":
    unittest.main()
