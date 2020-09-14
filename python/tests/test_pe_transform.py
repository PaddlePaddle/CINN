#!/usr/bin/env python3
import unittest
import cinn
import numpy as np
from cinn import runtime
from cinn import ir
from cinn import lang
from cinn import Target
from cinn import pe
from cinn.poly import create_stages
from cinn.common import *


class TestPETransform(unittest.TestCase):
    def setUp(self):
        self.m = 100
        self.n = 32
        self.k = 16

        self.target = Target()
        self.target.arch = Target.Arch.X86
        self.target.bits = Target.Bit.k64
        self.target.os = Target.OS.Linux

        self.transform_data = []

    def test_transform_0(self):
        for (fn_name, pe_fn, np_fn) in [
            ("matmul", pe.matmul, np.matmul),
        ]:
            self.compiler = cinn.Compiler.create(self.target)
            self.transform_matmul_tester(fn_name, pe_fn, np_fn, False, False,
                                         1, 1)

    def test_transform_1(self):
        for (fn_name, pe_fn, np_fn) in [
            ("matmul", pe.matmul, np.matmul),
        ]:
            self.compiler = cinn.Compiler.create(self.target)
            self.transform_matmul_tester(fn_name, pe_fn, np_fn, False, True, 1,
                                         1)

    def transform_matmul_tester(self, fn_name, cinn_fn, np_fn, trans_a,
                                trans_b, x_num_col_dims, y_num_col_dims):
        m, n, k = [ir.Expr(_) for _ in (
            self.m,
            self.n,
            self.k,
        )]
        x_shape_expr = [k, m] if trans_a else [m, k]
        y_shape_expr = [n, k] if trans_b else [k, n]
        x = lang.Placeholder("float32", "x", x_shape_expr)
        y = lang.Placeholder("float32", "y", y_shape_expr)
        func_name = "test_" + fn_name

        stages = create_stages([x.to_tensor(), y.to_tensor()])
        z = cinn_fn(x.to_tensor(), y.to_tensor(), stages, trans_a, trans_b,
                    x_num_col_dims, y_num_col_dims)
        temp_input = [x.to_tensor(), y.to_tensor()]
        for tensor in z:
            temp_input.append(tensor)

        func = lang.lower(func_name, stages, temp_input)
        print(func)

        builder = lang.Module.Builder("transform_module", self.target)
        builder.add_function(func)

        module = builder.build()
        self.compiler.build(module)

        fn = self.compiler.lookup(func_name)

        x_data, y_data, x_buf, y_buf, out_buf, *args = self.create_data(
            (self.m, self.n), trans_a, trans_b)
        fn(args)

        self.assertTrue(
            np.allclose(
                out_buf.numpy(),
                self.create_target_data(np_fn, x_data, y_data, trans_a,
                                        trans_b),
                atol=1e-4))

    def create_target_data(self, np_target_fn, x_data, y_data, trans_a,
                           trans_b):
        x_data = np.transpose(x_data) if trans_a else x_data
        y_data = np.transpose(y_data) if trans_b else y_data
        return np_target_fn(x_data, y_data)

    def create_data(self, output_shape, trans_a, trans_b):
        if not self.transform_data:
            if trans_a:
                x_data = np.around(
                    np.random.randn(self.k, self.m).astype("float32"), 2)
            else:
                x_data = np.around(
                    np.random.randn(self.m, self.k).astype("float32"), 2)
            if trans_b:
                y_data = np.around(
                    np.random.randn(self.n, self.k).astype("float32"), 2)
            else:
                y_data = np.around(
                    np.random.randn(self.k, self.n).astype("float32"), 2)
            x = runtime.cinn_buffer_t(x_data, runtime.cinn_x86_device)
            y = runtime.cinn_buffer_t(y_data, runtime.cinn_x86_device)
            out = runtime.cinn_buffer_t(
                np.zeros(output_shape).astype("float32"),
                runtime.cinn_x86_device)
            self.transform_data = [
                x_data, y_data, x, y, out,
                runtime.cinn_pod_value_t(x),
                runtime.cinn_pod_value_t(y),
                runtime.cinn_pod_value_t(out)
            ]

        return self.transform_data


if __name__ == "__main__":
    unittest.main()
