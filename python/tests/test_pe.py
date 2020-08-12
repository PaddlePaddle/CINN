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

        self.compiler = cinn.Compiler.create(self.target)

    def test_elementwise(self):
        m, n = [ir.Expr(_) for _ in (
            self.m,
            self.n,
        )]
        x = lang.Placeholder("float32", "x", [m, n])
        y = pe.tanh(x.to_tensor())

        func = lang.lower("test_tanh", [x.to_tensor(), y])

        builder = lang.Module.Builder("elementwise_module", self.target)
        builder.add_function(func)

        module = builder.build()
        self.compiler.build(module)

        fn = self.compiler.lookup("test_tanh")

        x_data, out_data, x_buf, out_buf, *args = self.create_data()
        fn(args)

        self.assertTrue(np.allclose(out_buf.numpy(), out_data, atol=1e-4))

    def create_data(self):
        x_data = np.around(
            np.random.randn(self.m, self.n).astype("float32"), 2)
        x = runtime.cinn_buffer_t(x_data, runtime.cinn_x86_device)
        out = runtime.cinn_buffer_t(
            np.zeros([self.m, self.n]).astype("float32"),
            runtime.cinn_x86_device)
        out_data = np.tanh(x_data)

        return [
            x_data, out_data, x, out,
            runtime.cinn_pod_value_t(x),
            runtime.cinn_pod_value_t(out)
        ]


if __name__ == "__main__":
    unittest.main()
