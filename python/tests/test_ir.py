#!/usr/bin/env python3
import unittest
import cinn
from cinn import runtime
from cinn import ir
from cinn import lang
from cinn.optim import *
from cinn import Target
from cinn.common import *
from cinn.ir import *


class TestIR(unittest.TestCase):
    def test_pod(self):
        one = Expr(1)
        self.assertEqual(str(simplify(one + one)), "2")
        self.assertEqual(str(simplify(one * Expr(0))), "0")

    def test_expr(self):
        a = Var("a")
        b = Var("b")

        expr = 1 + b
        print(expr)

        expr = b + 1
        print(expr)

        self.assertEqual(str(b * 0), "0")
        print(expr)
        print(simplify(expr))


if __name__ == "__main__":
    unittest.main()
