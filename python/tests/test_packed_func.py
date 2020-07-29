#!/usr/bin/env python3

import unittest
import numpy as np
import cinn
from cinn import ir
from cinn import CINNValue
from math import isclose


class TestPackedFunc(unittest.TestCase):
    def setUp(self):
        pass

    def test_lambda(self):
        add3 = ir.register_packed_func("test_packed_func_add3")(lambda x, y, z: x + y + z)
        self.assertEqual(add3(1, 2, 3), 6)
        self.assertEqual(ir.get_global_func("test_packed_func_add3"), add3)
        self.assertTrue(isinstance(add3, ir.PackedFunc))

    def test_normal_function(self):
        @ir.register_packed_func("test_packed_func_mul")
        def mul(x, y):
            return x * y

        self.assertTrue(isclose(mul(2.3, 3.0), 6.9, abs_tol=1e-5))
        self.assertEqual(mul(4, 5), 20)

    def test_callable_object(self):
        class Accumulator(object):
            def __init__(self, init):
                self.init = init

            def __call__(self, *args):
                r = cinn.CINNValue(self.init)
                for arg in args:
                    r = r + arg
                return r

        accumulate = ir.register_packed_func("accumulate_float")(Accumulator(1.0))
        self.assertTrue(isclose(accumulate(1., 2., 3., 4.), 11.))


if __name__ == "__main__":
    unittest.main()
