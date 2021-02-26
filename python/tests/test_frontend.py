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

assert len(sys.argv) == 1 + 2 + 1  # model and enable_gpu count
enable_gpu = sys.argv.pop()
multi_fc_model_dir = sys.argv.pop()
naive_model_dir = sys.argv.pop()


class TestFrontend(unittest.TestCase):
    def setUp(self):
        if enable_gpu == "ON":
            self.target = DefaultNVGPUTarget()
        else:
            self.target = DefaultHostTarget()

    def paddle_verify(self, result):
        paddle.enable_static()

        a = fluid.layers.data(name='A', shape=[24, 56, 56], dtype='float32')
        b = fluid.layers.data(name='B', shape=[24, 56, 56], dtype='float32')
        c = fluid.layers.elementwise_add(a, b)
        d = fluid.layers.relu(c)
        e = fluid.initializer.NumpyArrayInitializer(
            np.array(result[2]).reshape((144, 24, 1, 1)).astype("float32"))
        f = fluid.layers.conv2d(
            input=d,
            num_filters=144,
            filter_size=1,
            stride=1,
            padding=0,
            dilation=1,
            param_attr=e)
        f1 = fluid.layers.relu(f)
        g = fluid.layers.scale(f1, scale=2.0, bias=0.5)
        res = fluid.layers.softmax(g, axis=1)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        x = np.array(result[0]).reshape((2, 24, 56, 56)).astype("float32")
        y = np.array(result[1]).reshape((2, 24, 56, 56)).astype("float32")
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

        a = Variable("A").set_type(Float(32)).set_shape([2, 24, 56, 56])
        b = Variable("B").set_type(Float(32)).set_shape([2, 24, 56, 56])
        c = prog.add(a, b)
        d = prog.relu(c)
        e = Variable("E").set_type(Float(32)).set_shape([144, 24, 1, 1])
        f = prog.conv2d(d, e, {
            "stride": [1, 1],
            "dilation": [1, 1],
            "padding": [0, 0]
        })
        # Here in X86, if we delete this relu op, the result is correct.
        # But if we add relu op, the fused op of (conv2d+relu) will cause Segmentation fault.
        f1 = prog.relu(f)
        g = prog.scale(f1, {"scale": 2.0, "bias": 0.5})
        h = prog.softmax(g, {"axis": 1})

        # self.assertEqual(prog.size(), 6)
        # print program
        for i in range(prog.size()):
            print(prog[i])
        tensor_data = [
            np.random.random([2, 24, 56, 56]).astype("float32"),
            np.random.random([2, 24, 56, 56]).astype("float32"),
            np.random.random([144, 24, 1, 1]).astype("float32")
        ]

        result = prog.build_and_get_output(self.target, [a, b, e], tensor_data,
                                           h)
        result = result.numpy(self.target).reshape(-1)
        tensor_data.append(result)
        # self.paddle_verify(tensor_data)


if __name__ == "__main__":
    unittest.main()
