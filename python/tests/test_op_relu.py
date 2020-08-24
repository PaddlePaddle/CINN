#!/usr/bin/env python3
import cinn
from cinn import frontend
from cinn import runtime
from cinn import lang
from cinn import framework
from cinn import ir
from cinn import common
import unittest
import numpy as np


class TestOpRelu(unittest.TestCase):
    def test_op_add(self):
        a, c, c_target, *args = create_data()
        module = test_codegen()
        self.engine = cinn.ExecutionEngine()
        self.engine.link(module)
        unittest = self.engine.lookup("op_unittest")
        unittest(args)
        c_data = c.numpy()
        c_target_data = c_target.numpy()
        self.assertTrue(np.allclose(c_data, c_target_data, atol=1e-4))


def test_codegen():
    a = lang.Placeholder("float32", "A", [ir.Expr(100), ir.Expr(32)])
    # b = lang.Placeholder("float32", "B", [ir.Expr(100), ir.Expr(32)])
    inputs = [a.to_tensor()]
    types = [common.Float(32)]
    attrs = framework.NodeAttr()
    target = common.Target()
    target.arch = common.Target.Arch.X86
    target.bits = common.Target.Bit.k32
    target.os = common.Target.OS.Linux
    strategy_map = framework.Operator.get_op_attrs("CINNStrategy")
    res = strategy_map.apply_strategy("relu", attrs, inputs, types, target)
    func = lang.lower("op_unittest", res)
    print('func', func)
    builder = lang.Module.Builder("op_unittest", target)
    builder.add_function(func)
    return builder.build()


def create_data():
    a_init = np.around(np.random.randn(100, 32).astype("float32"), 2)
    # b_init = np.around(np.random.randn(100, 32).astype("float32"), 2)
    a = runtime.cinn_buffer_t(a_init, runtime.cinn_x86_device)
    # b = runtime.cinn_buffer_t(b_init, runtime.cinn_x86_device)
    c = runtime.cinn_buffer_t(
        np.zeros([100, 32]).astype("float32"), runtime.cinn_x86_device)
    c_target = runtime.cinn_buffer_t(np.maximum(a.numpy(),np.zeros([100, 32]).astype("float32")),
                                     runtime.cinn_x86_device)

    a_arg = runtime.cinn_pod_value_t(a)
    # b_arg = runtime.cinn_pod_value_t(b)
    c_arg = runtime.cinn_pod_value_t(c)
    return [a, c, c_target, a_arg, c_arg]


if __name__ == "__main__":
    unittest.main()
