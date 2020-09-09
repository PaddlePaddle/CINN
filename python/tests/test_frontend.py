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

    def test_model(self):
        x_shape = [20, 10]
        self.executor = Executor(["a"], [x_shape])
        self.executor.load_paddle_model(
            "/home/chunwei/project/cinn2/tests/fake_model/model2", False)
        a_t = self.executor.get_tensor("a")
        # a_t.from_numpy(np.random.random(x_shape))
        a_t.from_numpy(np.ones(x_shape, dtype="float"))
        self.executor.run()

        out = self.executor.get_tensor("fc_0.tmp_1")
        print("out", out.numpy())


if __name__ == "__main__":
    unittest.main()
