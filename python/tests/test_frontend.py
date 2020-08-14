#!/usr/bin/env python3
from cinn.frontend import *
import unittest
import cinn
import numpy as np


class TestFrontend(unittest.TestCase):
    def test_basic(self):
        prog = Program()
        a = Variable("a")
        b = Variable("b")
        c = prog.add(a, b)
        d = prog.add(c, b)

        self.assertEqual(prog.size(), 2)
        # print program
        for i in range(prog.size()):
            print(prog[i])


if __name__ == "__main__":
    unittest.main()
