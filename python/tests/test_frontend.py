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

    def paddle_test(self, result):
        print(
            "Do nothing now! Import paddle leads to segmentation fault. To be fixed."
        )

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

        self.assertEqual(prog.size(), 3)
        # print program
        for i in range(prog.size()):
            print(prog[i])
        result = prog.build_with_inputs(self.target, [a, b, e], f3)
        print("The result list's size is: ")
        print(len(result))
        print("The output tensor's length is: ")
        print(len(result[len(result) - 1]))
        self.paddle_test(result)


if __name__ == "__main__":
    unittest.main()
