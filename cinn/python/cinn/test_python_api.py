import os
import sys
import numpy as np
import unittest

# from cinn_core import cinn_core
import cinn_core as cinn


class TestElementwiseAdd(unittest.TestCase):
    def setUp(self) -> None:
        context = cinn.Context()

        comp = cinn.Computation(context, "fn0")
        p0_shape = cinn.Shape()
        p0_shape.add_int_dim(10)
        p0_shape.add_int_dim(20)

        p0 = comp.add_parameter(0, p0_shape, "p0", "float32")
        p1 = comp.add_parameter(1, p0_shape, "p1", "float32")

        add_out = comp.add_binary("add", p0, p1)

        module = cinn.Module("module0")
        module.add_entry_computation(comp)

        # compile it
        self.compiler = cinn.Compiler()
        self.compiler.compile(module)

    def test1(self):
        py_data0 = cinn.Buffer(np.float32(np.random.random((10, 20))))
        py_data1 = cinn.Buffer(np.float32(np.random.random((10, 20))))
        py_data2 = cinn.Buffer(np.zeros((10, 20), dtype='float32'))

        args = cinn.Args()
        args.add_buffer(py_data0)
        args.add_buffer(py_data1)
        args.add_buffer(py_data2)

        self.compiler.eval("fn0", args)

        np_data0 = py_data0.to_numpy()
        np_data1 = py_data1.to_numpy()
        np_data2 = py_data2.to_numpy()

        self.assertTrue((np_data2 == (np_data0 + np_data1)).all())

        print(py_data2.to_numpy())


if __name__ == '__main__':
    unittest.main()
