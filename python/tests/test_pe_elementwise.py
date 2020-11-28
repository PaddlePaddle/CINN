#!/usr/bin/env python3
import unittest
import cinn
import numpy as np
from cinn import runtime
from cinn import ir
from cinn.poly import create_stages
from cinn import lang
from cinn import Target
from cinn import pe
from cinn.common import *


class TestPEElementwise(unittest.TestCase):
    def setUp(self):
        self.m = 32
        self.n = 32

        self.target = Target()
        self.target.arch = Target.Arch.X86
        self.target.bits = Target.Bit.k32
        self.target.os = Target.OS.Linux

        self.unary_data = []

    def test_unary(self):
        for (fn_name, pe_fn, np_fn, dtype) in [
            ("exp", pe.exp, np.exp, "float32"),
                # TODO(wenming2014) not numpy
                # ("erf", pe.erf, np.erf, "float32"),
                # ("sqrt", pe.sqrt, np.sqrt, "float32"),
                # RuntimeWarning: divide by zero encountered in log2
                # ("log2", pe.log2, np.log2, "float32"),
                # ("log10", pe.log10, np.log10, "float32"),
            ("floor", pe.floor, np.floor, "float32"),
            ("ceil", pe.ceil, np.ceil, "float32"),
                # ("round", pe.round, np.round, "float32"),
            ("trunc", pe.trunc, np.trunc, "float32"),
            ("cos", pe.cos, np.cos, "float32"),
            ("cosh", pe.cosh, np.cosh, "float32"),
            ("tan", pe.tan, np.tan, "float32"),
            ("sin", pe.sin, np.sin, "float32"),
            ("sinh", pe.sinh, np.sinh, "float32"),
                # TODO(wenming2014) begin not numpy
                # ("acos", pe.acos, np.acos, "float32"),
                # ("acosh", pe.acosh, np.acosh, "float32"),
                # ("asin", pe.asin, np.asin, "float32"),
                # ("asinh", pe.asinh, np.asinh, "float32"),
                # ("atan", pe.atan, np.atan, "float32"),
                # ("atanh", pe.atanh, np.atanh, "float32"),
                # TODO(wenming2014) en
                # ("isnan", pe.isnan, np.isnan, "float32"),
            ("tanh", pe.tanh, np.tanh, "float32"),
                # ("isfinite", pe.isfinite, np.isfinite, "float32"),
                # ("isinf", pe.isinf, np.isinf, "float32"),
            ("negative", pe.negative, np.negative, "float32"),
                # ("identity", pe.identity, np.identity, "float32"),
                # TODO(wenming2014) int type
                # ("logical_not", pe.logical_not, np.logical_not, "int32"),
                # ("bitwise_not", pe.bitwise_not, np.bitwise_not, "int32"),
                # TODO(wenming2014) not numpy
                # ("sigmoid", pe.sigmoid, np.sigmoid, "float32"),
            ("sign", pe.sign, np.sign, "float32"),
            ("abs", pe.abs, np.abs, "float32"),
                # TODO(wenming2014) not numpy
                # ("rsqrt", pe.rsqrt, np.rsqrt, "float32"),
        ]:
            self.compiler = cinn.Compiler.create(self.target)
            self.union_tester(fn_name, pe_fn, np_fn, dtype)

    def union_tester(self, fn_name, cinn_fn, np_fn, dtype="float32"):
        m, n = [ir.Expr(_) for _ in (
            self.m,
            self.n,
        )]
        x = lang.Placeholder(dtype, "x", [m, n])
        y = cinn_fn(x.to_tensor())

        func_name = "test_" + fn_name

        stages = create_stages([x.to_tensor(), y])
        func = lang.lower(func_name, stages, [x.to_tensor(), y])

        builder = lang.Module.Builder("elementwise_module", self.target)
        builder.add_function(func)

        module = builder.build()
        self.compiler.build(module)

        fn = self.compiler.lookup(func_name)

        x_data, x_buf, out_buf, *args = self.create_data(dtype)
        fn(args)

        self.assertTrue(
            np.allclose(
                out_buf.numpy(),
                self.create_target_data(x_data, np_fn),
                atol=1e-4))

    def create_target_data(self, x_data, np_target_fn):
        return np_target_fn(x_data)

    def create_data(self, dtype):
        if not self.unary_data:
            x_data = np.around(
                np.random.randn(self.m, self.n).astype(dtype), 2)
            x = runtime.cinn_buffer_t(x_data, runtime.cinn_x86_device)
            out = runtime.cinn_buffer_t(
                np.zeros([self.m, self.n]).astype(dtype),
                runtime.cinn_x86_device)
            self.unary_data = [
                x_data, x, out,
                runtime.cinn_pod_value_t(x),
                runtime.cinn_pod_value_t(out)
            ]

        return self.unary_data


if __name__ == "__main__":
    unittest.main()
