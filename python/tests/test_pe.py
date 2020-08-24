#!/usr/bin/env python3
import unittest
import cinn
import numpy as np
from cinn import runtime
from cinn import ir
from cinn import lang
from cinn import Target
from cinn import pe
from cinn.poly import *
from cinn.common import *


class TestPE(unittest.TestCase):
    def setUp(self):
        self.m = 32
        self.n = 32

        self.target = Target()
        self.target.arch = Target.Arch.X86
        self.target.bits = Target.Bit.k32
        self.target.os = Target.OS.Linux

        self.unary_data = []

    def test_unary(self):
        for (fn_name, pe_fn, np_fn) in [
            ("tanh", pe.tanh, np.tanh),
            ("cos", pe.cos, np.cos),
            ("exp", pe.exp, np.exp),
        ]:
            self.compiler = cinn.Compiler.create(self.target)
            self.union_tester(fn_name, pe_fn, np_fn)

    def union_tester(self, fn_name, cinn_fn, np_fn):
        m, n = [ir.Expr(_) for _ in (
            self.m,
            self.n,
        )]
        x = lang.Placeholder("float32", "x", [m, n])
        y = cinn_fn(x.to_tensor())

        func_name = "test_" + fn_name

        stages = create_stages([y])
        func = lang.lower(func_name, stages, [x.to_tensor(), y])

        print('C code', func)

        builder = lang.Module.Builder("elementwise_module", self.target)
        builder.add_function(func)

        module = builder.build()
        self.compiler.build(module)

        fn = self.compiler.lookup(func_name)

        x_data, x_buf, out_buf, *args = self.create_data()
        fn(args)

        self.assertTrue(
            np.allclose(
                out_buf.numpy(),
                self.create_target_data(x_data, np_fn),
                atol=1e-4))

    def create_target_data(self, x_data, np_target_fn):
        return np_target_fn(x_data)

    def create_data(self):
        if not self.unary_data:
            x_data = np.around(
                np.random.randn(self.m, self.n).astype("float32"), 2)
            x = runtime.cinn_buffer_t(x_data, runtime.cinn_x86_device)
            out = runtime.cinn_buffer_t(
                np.zeros([self.m, self.n]).astype("float32"),
                runtime.cinn_x86_device)
            self.unary_data = [
                x_data, x, out,
                runtime.cinn_pod_value_t(x),
                runtime.cinn_pod_value_t(out)
            ]

        return self.unary_data


if __name__ == "__main__":
    unittest.main()
