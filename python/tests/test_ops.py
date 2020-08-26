#!/usr/bin/env python3
import unittest
import numpy as np
import cinn
from cinn import frontend
from cinn import runtime
from cinn import lang
from cinn import framework
from cinn import ir
from cinn import common
from cinn.poly import create_stages


class TestOps(unittest.TestCase):
    def setUp(self):
        self.counter = 0
        self.target = common.Target()
        self.target.arch = common.Target.Arch.X86
        self.target.bits = common.Target.Bit.k32
        self.target.os = common.Target.OS.Linux

    def test_ops(self):
        print("Test for op add begin:")
        self.op_unittest([[100, 32], [100, 32]], [100, 32], "add")
        print("Test for op relu begin:")
        self.op_unittest([[32, 32]], [32, 32], "relu")

    def op_unittest(self, input_shapes, output_shape, op_name):
        self.compiler = cinn.Compiler.create(self.target)
        inputs = []
        inputs_data = []
        for i_shape in input_shapes:
            expr_shape = []
            inputs_data.append(
                np.around(np.random.random(i_shape).astype("float32"), 3))
            for dim_shape in i_shape:
                expr_shape.append(ir.Expr(dim_shape))
            inputs.append(
                lang.Placeholder("float32", self.GetName(),
                                 expr_shape).to_tensor())
        module = self.codegen(op_name, inputs)
        self.compiler.build(module)
        fn = self.compiler.lookup(op_name)
        out = runtime.cinn_buffer_t(
            np.zeros(output_shape).astype("float32"), runtime.cinn_x86_device)
        args = []
        temp_inputs = []
        for in_data in inputs_data:
            temp_inputs.append(
                runtime.cinn_buffer_t(in_data, runtime.cinn_x86_device))
        for in_data in temp_inputs:
            args.append(runtime.cinn_pod_value_t(in_data))
        args.append(runtime.cinn_pod_value_t(out))
        fn(args)
        self.assertTrue(
            np.allclose(
                out.numpy(),
                self.create_target_data(inputs_data, op_name),
                atol=1e-4))

    def codegen(self, op_name, inputs):
        types = [common.Float(32)]
        attrs = framework.NodeAttr()
        strategy_map = framework.Operator.get_op_attrs("CINNStrategy")
        res = strategy_map.apply_strategy(op_name, attrs, inputs, types,
                                          self.target)
        stages = create_stages(res)
        func = lang.lower(op_name, stages, res)
        print('func', func)
        builder = lang.Module.Builder(op_name, self.target)
        builder.add_function(func)
        return builder.build()

    def create_target_data(self, inputs_data, op_name):
        if (op_name == "add"):
            X, Y = inputs_data
            return X + Y
        elif (op_name == "relu"):
            X = inputs_data
        return np.maximum(X, np.zeros(np.array(X).shape).astype("float32"))

    def GetName(self):
        self.counter = self.counter + 1
        return "Var_" + str(self.counter)


if __name__ == "__main__":
    print("test begin!!")
    unittest.main()
