#!/usr/bin/env python3

import unittest
import numpy as np
import cinn
from cinn import runtime
from cinn import ir
from cinn import lang
from cinn import Target
from cinn.poly import create_stages


class TestMamul(unittest.TestCase):
    def setUp(self):
        self.target = Target()
        self.target.arch = Target.Arch.X86
        self.target.bits = Target.Bit.k32
        self.target.os = Target.OS.Linux
        self.m = 1024
        self.n = 1024
        self.k = 1024
        self.bn = 32

        self.engine = cinn.ExecutionEngine()

    def test_matmul_basic(self):
        a, b, c, c_target, *args = create_data(self.m, self.n, self.k, self.bn)
        module = create_matmul_basic(self.target, self.m, self.n, self.k)

        self.engine.link(module)
        matmul = self.engine.lookup("matmul")
        matmul(args)
        cd = c.numpy()
        cd_target = c_target.numpy()
        self.assertTrue(np.allclose(cd, cd_target, atol=1e-4))

    def test_matmul_tile(self):
        a, b, c, c_target, *args = create_data(self.m, self.n, self.k, self.bn)
        module = create_matmul_tile(self.target, self.m, self.n, self.k)
        print('module:\n', module.get_c_code())
        self.engine.link(module)
        matmul = self.engine.lookup("matmul_tile")
        matmul(args)
        cd = c.numpy()
        cd_target = c_target.numpy()
        self.assertTrue(np.allclose(cd, cd_target, atol=1e-4))


def create_matmul_basic(target, m, n, k):
    m, n, k = [ir.Expr(_) for _ in (m, n, k)]

    a = lang.Placeholder("float32", "A", [m, k])
    b = lang.Placeholder("float32", "B", [k, n])

    c_init = lang.compute([m, n], lambda v: ir.Expr(0.), "c_init")

    k1 = ir.Var(k.as_int32(), "k1")
    c = lang.compute([m, n], lambda v: lang.sum(
        a(v[0], k1.to_expr_mutable()) * b(k1.to_expr_mutable(), v[1])), "c",
                     [k1])

    stages = create_stages([c_init, c])
    c_stage = stages[c]
    stages[c_init].share_buffer_with(c_stage)
    stages[c].ctrl_depend(c_init)

    builder = lang.Module.Builder("matmul", target)

    ts = [a.to_tensor(), b.to_tensor(), c_init, c]
    func = lang.lower("matmul", stages, ts)
    print('func', func)
    builder.add_function(func)
    return builder.build()


def create_matmul_tile(target, m, n, k):
    m, n, k = [ir.Expr(_) for _ in [m, n, k]]
    a = lang.Placeholder("float32", "A", [m, k])
    b = lang.Placeholder("float32", "B", [k, n])

    c_init = lang.compute([m, n], lambda v: ir.Expr(0.), "c_init")

    k1 = ir.Var(k.as_int32(), "k1")
    c = lang.compute([m, n], lambda v: lang.sum(
        a(v[0], k1.to_expr_mutable()) * b(k1.to_expr_mutable(), v[1])), "c",
                     [k1])

    stages = create_stages([c_init, c])
    stages[c].share_buffer_with(stages[c_init])
    stages[c].ctrl_depend(c_init)
    stages[c].tile(0, 1, 4, 4)

    builder = lang.Module.Builder("matmul_tile", target)
    ts = [a.to_tensor(), b.to_tensor(), c_init, c]
    func = lang.lower("matmul_tile", stages, ts)
    print('func', func)
    builder.add_function(func)
    return builder.build()


def create_data(m, n, k, bn):
    # call around to lower the numpy's float precision so that it will not vary too much from C's float precision.
    a_init = np.around(np.random.randn(m, k).astype("float32"), 2)
    b_init = np.around(np.random.randn(k, n).astype("float32"), 2)
    a = runtime.cinn_buffer_t(a_init, runtime.cinn_x86_device)
    b = runtime.cinn_buffer_t(b_init, runtime.cinn_x86_device)
    c = runtime.cinn_buffer_t(
        np.zeros([m, n]).astype("float32"), runtime.cinn_x86_device)
    c_target = runtime.cinn_buffer_t(a.numpy() @ b.numpy(),
                                     runtime.cinn_x86_device)
    packed_b = runtime.cinn_buffer_t(
        np.zeros([n // bn, k, bn]).astype("float32"), runtime.cinn_x86_device)

    a_arg = runtime.cinn_pod_value_t(a)
    b_arg = runtime.cinn_pod_value_t(b)
    c_arg = runtime.cinn_pod_value_t(c)
    packed_b_arg = runtime.cinn_pod_value_t(packed_b)
    return [a, b, c, c_target, a_arg, b_arg, c_arg]


if __name__ == "__main__":
    unittest.main()
