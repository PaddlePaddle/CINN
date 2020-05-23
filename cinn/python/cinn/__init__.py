import cinn_core
from cinn_core import Context, Computation, Compiler, Buffer, Module

def Shape(li):
    data = cinn_core.Shape()
    for v in li:
        if type(v) is int:
            data.add_int_dim(v)
        elif type(v) is str:
            data.add_var_dim(v)
        else:
            assert False
    return data

def Args(li):
    data = cinn_core.Args()
    for v in li:
        if type(v) is cinn_core.Buffer:
            data.add_buffer(v)
        elif type(v) is int:
            data.add_int32(v)
        else:
            assert False

    return data
