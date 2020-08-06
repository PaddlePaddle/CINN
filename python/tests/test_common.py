#!/usr/bin/env python3
import unittest
import cinn
from cinn import runtime
from cinn import ir
from cinn import lang
from cinn import Target
from cinn.common import *


class TestType(unittest.TestCase):
    def test_type_constructs(self):
        self.assertEqual(str(Float(32)), "float32")
        self.assertEqual(str(Int(32)), "int32")
        self.assertEqual(str(Int(64)), "int64")
        self.assertEqual(str(UInt(64)), "uint64")
        self.assertEqual(str(UInt(32)), "uint32")
        self.assertEqual(str(Bool()), "uint1")


if __name__ == "__main__":
    unittest.main()
