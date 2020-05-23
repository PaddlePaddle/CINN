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
        p0_shape.add_int_dim(100)
        p0_shape.add_int_dim(20)

        p0 = comp.add_parameter(0, p0_shape, "p0", "float32")
        p1 = comp.add_parameter(1, p0_shape, "p1", "float32")

        add_out = comp.add(p0, p1)

        module = cinn.Module("module0")
        module.add_entry_computation(comp)

        # compile it
        self.compiler = cinn.Compiler()
        self.compiler.compile(module)

    def test1(self):
        py_data0 = cinn.Buffer(np.float32(np.random.random((100, 20))))
        py_data1 = cinn.Buffer(np.float32(np.random.random((100, 20))))
        py_data2 = cinn.Buffer(np.zeros((100, 20), dtype='float32'))

        args = cinn.Args()
        args.add_buffer(py_data0)
        args.add_buffer(py_data1)
        args.add_buffer(py_data2)

        self.compiler.eval("fn0", args)

        np_data0 = py_data0.to_numpy()
        np_data1 = py_data1.to_numpy()
        np_data2 = py_data2.to_numpy()

        self.assertTrue((np_data2 == (np_data0 + np_data1)).all())

class TestDot(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 20
        self.M = 100
        self.N = 30
        self.K = 50

        context = cinn.Context()

        comp = cinn.Computation(context, "fn0")
        x_shape = cinn.Shape()
        x_shape.add_var_dim("n")
        x_shape.add_int_dim(self.M)
        x_shape.add_int_dim(self.K)

        w_shape = cinn.Shape()
        w_shape.add_int_dim(self.K)
        w_shape.add_int_dim(self.N)

        xp = comp.add_parameter(0, x_shape, "x", "float32")
        wp = comp.add_parameter(1, w_shape, "w", "float32")
        dot_out = comp.add_dot(xp, wp)

        module = cinn.Module("module0")
        module.add_entry_computation(comp)

        self.compiler = cinn.Compiler()
        self.compiler.compile(module)

    def test(self):
        py_data0 = cinn.Buffer(np.float32(np.random.random((self.batch_size, self.M, self.K))))
        py_data1 = cinn.Buffer(np.float32(np.random.random((self.K, self.N))))
        py_data2 = cinn.Buffer(np.zeros((self.batch_size, self.M, self.N), dtype='float32'))

        args = cinn.Args()
        args.add_int32(self.batch_size)
        args.add_buffer(py_data0)
        args.add_buffer(py_data1)
        args.add_buffer(py_data2)

        self.compiler.eval("fn0", args)

        np_data0 = py_data0.to_numpy()
        np_data1 = py_data1.to_numpy()
        np_data2 = py_data2.to_numpy()

        target = np_data0.dot(np_data1)

        self.assertTrue(np.isclose(np_data2 , target).all())



if __name__ == '__main__':
    unittest.main()
