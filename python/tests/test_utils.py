#!/usr/bin/env python3
import unittest
import math
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
        np.random.seed(0)
        self.counter = 0
        self.target = common.Target()
        self.target.arch = common.Target.Arch.X86
        self.target.bits = common.Target.Bit.k32
        self.target.os = common.Target.OS.Linux

    def create_target_data(self, inputs_data, attrs):
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

    def to_test_op(self,
                   input_shapes,
                   output_shapes,
                   op_name,
                   attrs,
                   out_index=None,
                   do_infer_shape=False):
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

        args = []
        temp_inputs = []
        alignment = 0
        if self.target.arch == common.Target.Arch.X86:
            alignment = 32
        for in_data in inputs_data:
            temp_inputs.append(
                runtime.cinn_buffer_t(in_data, runtime.cinn_x86_device,
                                      alignment))
        for in_data in temp_inputs:
            args.append(runtime.cinn_pod_value_t(in_data))
        if output_shapes == None:
            correct_result, output_shapes = self.create_target_data(
                inputs_data, attrs)
        else:
            correct_result = self.create_target_data(inputs_data, attrs)

        module = self.__codegen(op_name, inputs, output_shapes, attrs)

        self.compiler.build(module)
        fn = self.compiler.lookup(op_name)

        out = []

        for out_shape in output_shapes:
            out.append(
                runtime.cinn_buffer_t(
                    np.zeros(out_shape).astype("float32"),
                    runtime.cinn_x86_device, alignment))
        if do_infer_shape:
            infer_shapes = framework.Operator.get_op_shape_attrs("infershape")
            out_shapes = infer_shapes.infer_shape(op_name, input_shapes,
                                                  attrs.attr_store)
            print("out_shapes", out_shapes)
            for out_shape in out_shapes[1:]:
                out.append(
                    runtime.cinn_buffer_t(
                        np.zeros(out_shape).astype("float32"),
                        runtime.cinn_x86_device, alignment))

        for out_data in out:
            args.append(runtime.cinn_pod_value_t(out_data))
        fn(args)

        out_result = out[len(out) - 1].numpy()
        if out_index != None:
            out_result = out[out_index].numpy()
        self.assertTrue(np.allclose(out_result, correct_result, atol=1e-4))

    def __codegen(self, op_name, inputs, output_shapes, attrs):
        types = [common.Float(32)]
        strategy_map = framework.Operator.get_op_attrs("CINNStrategy")
        func = strategy_map.apply_strategy(op_name, attrs, inputs, types,
                                           output_shapes, self.target)
        logging.warning('func:\n\n%s\n', func)
        builder = lang.Module.Builder(op_name, self.target)
        builder.add_function(func)
        return builder.build()

    def __gen_var_name(self):
        self.counter = self.counter + 1
        return "Var_" + str(self.counter)


if __name__ == "__main__":
    unittest.main()
