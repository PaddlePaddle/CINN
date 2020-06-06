import os
import sys
import numpy as np
import unittest

import cinn


class TestElementwiseAdd(unittest.TestCase):
    def setUp(self) -> None:
        context = cinn.Context()

        comp = cinn.Computation(context, "fn0")

        p0_shape = cinn.Shape([100, 20])

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

        args = cinn.Args([py_data0, py_data1, py_data2])

        self.compiler.eval("fn0", args)

        np_data0 = py_data0.numpy()
        np_data1 = py_data1.numpy()
        np_data2 = py_data2.numpy()

        self.assertTrue((np_data2 == (np_data0 + np_data1)).all())


class TestDot(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 20
        self.M = 100
        self.N = 30
        self.K = 50

        context = cinn.Context()

        comp = cinn.Computation(context, "fn0")
        x_shape = cinn.Shape(["n", self.M, self.K])
        w_shape = cinn.Shape([self.K, self.N])

        xp = comp.add_parameter(0, x_shape, "x", "float32")
        wp = comp.add_parameter(1, w_shape, "w", "float32")
        dot_out = comp.dot(xp, wp)

        module = cinn.Module("module0")
        module.add_entry_computation(comp)

        self.compiler = cinn.Compiler()
        self.compiler.compile(module)

    def test(self):
        py_data0 = cinn.Buffer(np.float32(np.random.random((self.batch_size, self.M, self.K))))
        py_data1 = cinn.Buffer(np.float32(np.random.random((self.K, self.N))))
        py_data2 = cinn.Buffer(np.zeros((self.batch_size, self.M, self.N), dtype='float32'))

        args = cinn.Args([self.batch_size, py_data0, py_data1, py_data2])

        self.compiler.eval("fn0", args)

        np_data0 = py_data0.numpy()
        np_data1 = py_data1.numpy()
        np_data2 = py_data2.numpy()

        target = np_data0.dot(np_data1)

        self.assertTrue(np.isclose(np_data2, target).all())


class TestElementwise(unittest.TestCase):
    def setUp(self):
        self.batch_size = 20
        self.M = 100

        self.context = cinn.Context()
        self.shape = cinn.Shape(["n", self.M])

        self.module = cinn.Module("module0")

        self.create_add_comp()
        self.create_sub_comp()
        self.create_div_comp()
        self.create_mul_comp()

        self.create_elementwise_comp("tanh0", lambda x: x.tanh)
        self.create_elementwise_comp("ceil0", lambda x: x.ceil)
        self.create_elementwise_comp("abs0", lambda x: x.abs)
        # self.create_elementwise_comp("floor0", lambda x : x.floor)

        self.create_complex_inline_comp()

        self.compiler = cinn.Compiler()
        self.compiler.compile(self.module)

        self.py_data0 = cinn.Buffer(np.float32(np.random.random((self.batch_size, self.M))))
        self.py_data1 = cinn.Buffer(np.float32(np.random.random((self.batch_size, self.M))))
        self.py_out = cinn.Buffer(np.zeros((self.batch_size, self.M), dtype='float32'))

    def create_add_comp(self):
        comp = cinn.Computation(self.context, "add0")

        x = comp.add_parameter(0, self.shape, "x", "float32")
        y = comp.add_parameter(1, self.shape, "y", "float32")
        out = comp.add(x, y)

        self.module.add_computation(comp)

    def create_sub_comp(self):
        comp = cinn.Computation(self.context, "sub0")

        x = comp.add_parameter(0, self.shape, "x", "float32")
        y = comp.add_parameter(1, self.shape, "y", "float32")
        out = comp.sub(x, y)

        self.module.add_computation(comp)

    def create_mul_comp(self):
        comp = cinn.Computation(self.context, "mul0")

        x = comp.add_parameter(0, self.shape, "x", "float32")
        y = comp.add_parameter(1, self.shape, "y", "float32")
        out = comp.mul(x, y)

        self.module.add_computation(comp)

    def create_div_comp(self):
        return
        comp = cinn.Computation(self.context, "div0")

        x = comp.add_parameter(0, self.shape, "x", "float32")
        y = comp.add_parameter(1, self.shape, "y", "float32")
        out = comp.div(x, y)

        self.module.add_computation(comp)

    def create_tanh_comp(self):
        comp = cinn.Computation(self.context, "tanh0")

        x = comp.add_parameter(0, self.shape, "x", "float32")
        out = comp.tanh(x)

        self.module.add_computation(comp)

    def create_elementwise_comp(self, repr, fn):
        ''' fn: lambda comp: comp.tanh '''
        comp = cinn.Computation(self.context, repr)

        x = comp.add_parameter(0, self.shape, "x", "float32")
        out = fn(comp)(x)

        self.module.add_computation(comp)

    def create_complex_inline_comp(self):
        comp = cinn.Computation(self.context, "complex_inlined_")

        x = comp.add_parameter(0, self.shape, "x", "float32")
        y = comp.add_parameter(1, self.shape, "y", "float32")

        add_out = comp.add(x, y)
        tanh_out = comp.tanh(add_out)
        add1_out = comp.add(tanh_out, x)
        add2_out = comp.add(tanh_out, y)
        final_out = comp.mul(add1_out, add2_out)

        self.module.add_computation(comp)

    def create_conv(self):
        comp = cinn.Computation(self.context, "conv0")
        self.conv_shape_I_raw = [1, 200, 200, 3]
        self.conv_shape_W_raw = [4, 4, 3, 3]

        self.conv_shape_I = cinn.Shape(self.conv_shape_I_raw)
        self.conv_shape_W = cinn.Shape(self.conv_shape_W_raw)
        I = comp.add_parameter(0, self.conv_shape_I, "I", "float32")
        W = comp.add_parameter(1, self.conv_shape_W, "W", "float32")

        conv_out = comp.conv(I, W, 2, 2, 1, 1)

        self.conv_out_shape = [self.conv_shape_I_raw[0], self.conv_shape_W_raw[0],
                               (self.conv_shape_I_raw[2] - self.conv_shape_W_raw[2] + 2 * 2) / 1 + 1,
                               (self.conv_shape_I_raw[3] - self.conv_shape_W_raw[3] + 2 * 2) / 1 + 1]

        self.module.add_computation(comp)


    def test_add(self):
        args = cinn.Args([self.batch_size, self.py_data0, self.py_data1, self.py_out])
        self.compiler.eval("add0", args)

        self.assertTrue(np.isclose(self.py_data0.numpy() + self.py_data1.numpy(), self.py_out.numpy()).all())

    def test_sub(self):
        args = cinn.Args([self.batch_size, self.py_data0, self.py_data1, self.py_out])
        self.compiler.eval("sub0", args)

        self.assertTrue(np.isclose(self.py_data0.numpy() - self.py_data1.numpy(), self.py_out.numpy()).all())

    def test_mul(self):
        args = cinn.Args([self.batch_size, self.py_data0, self.py_data1, self.py_out])
        self.compiler.eval("mul0", args)

        self.assertTrue(np.isclose(self.py_data0.numpy() * self.py_data1.numpy(), self.py_out.numpy()).all())

    def test_div(self):
        return
        args = cinn.Args([self.batch_size, self.py_data0, self.py_data1, self.py_out])
        self.compiler.eval("div0", args)

        self.assertTrue(np.isclose(self.py_data0.numpy() * self.py_data1.numpy(), self.py_out.numpy()).all())

    def test_elementwise(self):
        args = cinn.Args([self.batch_size, self.py_data0, self.py_out])

        for fn_name, np_fn in [("tanh0", np.tanh),
                               ("ceil0", np.ceil),
                               ("abs0", np.abs),
                               ]:
            self.compiler.eval(fn_name, args)
            self.assertTrue(np.isclose(np_fn(self.py_data0.numpy()), self.py_out.numpy()).all())

    def test_complex_inline(self):
        args = cinn.Args([self.batch_size, self.py_data0, self.py_data1, self.py_out])

        self.compiler.eval("complex_inlined_", args)


        x = self.py_data0.numpy();
        y = self.py_data1.numpy();

        tanh_out = np.tanh(x+y)
        add1_out = x + tanh_out
        add2_out = y + tanh_out
        final_out = add1_out * add2_out

        self.assertTrue(np.allclose(self.py_out.numpy(), final_out))

    def test_conv(self):
        return
        I_data = cinn.Buffer(np.float32(np.random.random(self.conv_shape_I_raw)))
        W_data = cinn.Buffer(np.float32(np.random.random(self.conv_shape_W_raw)))
        py_out = cinn.Buffer(np.zeros((100, 20), dtype='float32'))


if __name__ == '__main__':
    unittest.main()
