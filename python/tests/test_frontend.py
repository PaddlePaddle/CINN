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
import paddle.fluid as fluid
import sys

model_dir = sys.argv.pop()


class TestFrontend(unittest.TestCase):
    def setUp(self):
        self.target = Target()
        self.target.arch = Target.Arch.X86
        self.target.bits = Target.Bit.k64
        self.target.os = Target.OS.Linux

    def test_basic(self):
        prog = Program()

        a = Variable("a").set_type(Float(32)).set_shape([1, 24, 56, 56])
        b = Variable("b").set_type(Float(32)).set_shape([1, 24, 56, 56])
        c = prog.add(a, b)
        d = prog.relu(c)
        e = Variable("e").set_type(Float(32)).set_shape([144, 24, 1, 1])
        [f1, f2, f3] = prog.conv2d(d, e, {
            "stride": [1, 1],
            "dilation": 1,
            "padding": [0, 0]
        })
        g = Variable("g").set_type(Float(32)).set_shape([4, 144])
        h = prog.batchnorm(f3, g, {"epsilon": 0.00001})

        self.assertEqual(prog.size(), 4)
        # print program
        for i in range(prog.size()):
            print(prog[i])
        prog.print_func(self.target)


class TestLoadPaddleModel(unittest.TestCase):
    def setUp(self):
        self.target = Target()
        self.target.arch = Target.Arch.X86
        self.target.bits = Target.Bit.k64
        self.target.os = Target.OS.Linux

        self.model_dir = model_dir

        self.x_shape = [4, 30]

    def get_paddle_inference_result(self, data):
        exe = fluid.Executor(fluid.CPUPlace())

        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(
             dirname=self.model_dir, executor=exe)

        results = exe.run(
            inference_program,
            feed={feed_target_names[0]: data},
            fetch_list=fetch_targets)

        result = results[0]
        return result

    def test_model(self):
        x_data = np.random.random(self.x_shape).astype("float32")
        self.executor = Executor(["a"], [self.x_shape])
        self.executor.load_paddle_model(self.model_dir, False)
        a_t = self.executor.get_tensor("a")
        a_t.from_numpy(x_data)

        out = self.executor.get_tensor("fc_0.tmp_2")
        out.from_numpy(np.zeros(out.shape(), dtype='float32'))

        self.executor.run()

        out = out.numpy()
        target_result = self.get_paddle_inference_result(x_data)

        print("out", out)
        self.assertTrue(np.allclose(out, target_result, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
