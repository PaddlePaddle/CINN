#!/usr/bin/env python3
import unittest
import cinn
import numpy as np
from cinn import runtime
from cinn import ir
from cinn import lang
from cinn import Target
from cinn import pe
from cinn.common import *


class TestPE(unittest.TestCase):
    def setUp(self):
        self.m = 256
        self.n = 256

        self.target = Target()
        self.target.arch = Target.Arch.X86
        self.target.bits = Target.Bit.k32
        self.target.os = Target.OS.Linux

        self.compiler = cinn.Compiler(self.target)

    def test_elementwise(self):
        m, n = [ir.Expr(_) for _ in (
            self.m,
            self.n,
        )]
        x = lang.Placeholder("float32", "x", [m, n])
        y = pe.tanh(x)

        func = lang.lower("test_tanh", [x.to_tensor(), y])

        builder = lang.Module.Builder("elementwise_module", target)
        builder.add_function(func)

        module = builder.build()
        self.compiler.build(module)

        # fn = self.compiler.lookup("test_tanh")

    def create_data():
        x_data = np.around(
            np.random.randn(self.m, self.n).astype("float32"), 2)
        x = runtime.cinn_buffer_t(x_data, runtime.cinn_x86_device)
        out = runtime.cinn_buffer_t(
            np.zeros(self.m, self.n).astype("float32"),
            runtime.cinn_x86_device)

        return [x, out]


if __name__ == "__main__":
    unittest.main()
