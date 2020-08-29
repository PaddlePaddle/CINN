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
import logging


class SingleOpTester(unittest.TestCase):
    '''
    A unittest framework for testing a single operator.

    Two methods one should override for each Operator's unittest

    1. create_target_data
    2. test_op
    '''

    def setUp(self):
        self.counter = 0
        self.target = common.Target()
        self.target.arch = common.Target.Arch.X86
        self.target.bits = common.Target.Bit.k32
        self.target.os = common.Target.OS.Linux

    def create_target_data(self, inputs_data):
        '''
        create the target of the operator's execution output.
        '''
        raise NotImplemented

    def test_op(self):
        '''
        USER API

        The real use case should implement this method!
        '''
        pass

    def to_test_op(self, input_shapes, output_shape, op_name):
        '''
        Test the operator.
        '''
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
                lang.Placeholder("float32", self.__gen_var_name(),
                                 expr_shape).to_tensor())
        module = self.__codegen(op_name, inputs)
        self.compiler.build(module)
        fn = self.compiler.lookup(op_name)
        out = []
        for out_shape in output_shape:
            out.append(
                runtime.cinn_buffer_t(
                    np.zeros(out_shape).astype("float32"),
                    runtime.cinn_x86_device))

        args = []
        temp_inputs = []
        for in_data in inputs_data:
            temp_inputs.append(
                runtime.cinn_buffer_t(in_data, runtime.cinn_x86_device))
        for in_data in temp_inputs:
            args.append(runtime.cinn_pod_value_t(in_data))
        for out_data in out:
            args.append(runtime.cinn_pod_value_t(out_data))

        fn(args)
        print("test op output is:")
        out_result = out[len(out) - 1]
        print(out_result.numpy())
        self.assertTrue(
            np.allclose(
                out_result.numpy(),
                self.create_target_data(inputs_data),
                atol=1e-4))

    def __codegen(self, op_name, inputs):
        types = [common.Float(32)]
        attrs = framework.NodeAttr()
        attrs.attr_store = {"test_attr_store": 1}
        attrs.set_attr("test_set_attr", 2)
        strategy_map = framework.Operator.get_op_attrs("CINNStrategy")
        res = strategy_map.apply_strategy(op_name, attrs, inputs, types,
                                          self.target)
        stages = create_stages(res)
        func = lang.lower(op_name, stages, res)
        logging.warning('func:\n\n%s\n', func)
        builder = lang.Module.Builder(op_name, self.target)
        builder.add_function(func)
        return builder.build()

    def __gen_var_name(self):
        self.counter = self.counter + 1
        return "Var_" + str(self.counter)


class OpTest_add(SingleOpTester):
    def create_target_data(self, inputs_data):
        X, Y = inputs_data
        return X + Y

    def test_op(self):
        self.to_test_op([[100, 32], [100, 32]], [[100, 32]], "add")


class OpTest_relu(SingleOpTester):
    def create_target_data(self, inputs_data):
        X = inputs_data
        return np.maximum(X, np.zeros(np.array(X).shape).astype("float32"))

    def test_op(self):
        self.to_test_op([[32, 32]], [[32, 32]], "relu")


class OpTest_conv2d(SingleOpTester):
    def create_target_data(self, inputs_data):
        return np.ones((1, 2, 5, 5)).astype("float32")

    def test_op(self):
        self.to_test_op([[1, 3, 10, 10], [2, 3, 2, 2]],
                        [[1, 3, 12, 12], [2, 3, 3, 3], [1, 2, 5, 5]], "conv2d")


if __name__ == "__main__":
    unittest.main()
