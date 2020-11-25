"""
JIT in CINN
=====================

In this tutorial, we will introduce the JIT module that execute the DSL on X86 and NV GPU.
"""

import cinn
import numpy as np
from cinn import runtime

##################################################################
# declare some variables for latter use
m = cinn.Expr(64)
n = cinn.Expr(64)
k = cinn.Expr(8)
bn = cinn.Expr(32)

##################################################################
# Decleare the computation
# -------------------------
A = cinn.Placeholder("float32", "A", [m, k])
B = cinn.Placeholder("float32", "B", [k, n])

kr = cinn.Var(k.as_int32(), "kr")
C = cinn.compute([
    m, n
], lambda v: cinn.reduce_sum(A(v[0], kr.expr()) * B(kr.expr(), v[1]), [kr]),
                 "C")

stages = cinn.create_stages([C])

target = cinn.Target()
builder = cinn.Module.Builder("matmul", target)

func = cinn.lower("matmul", stages, [A.to_tensor(), B.to_tensor(), C])
builder.add_function(func)
module = builder.build()

##################################################################
# Create a JIT engine.
# ---------------------
#
jit = cinn.ExecutionEngine()
jit.link(module)

##################################################################
# Execute the compiled function
#
a = runtime.cinn_buffer_t(
    np.random.randn(m.int(), k.int()).astype("float32"),
    runtime.cinn_x86_device)
b = runtime.cinn_buffer_t(
    np.random.randn(m.int(), k.int()).astype("float32"),
    runtime.cinn_x86_device)
c = runtime.cinn_buffer_t(
    np.zeros([m.int(), n.int()]).astype("float32"), runtime.cinn_x86_device)

args = [runtime.cinn_pod_value_t(_) for _ in [a, b, c]]
matmul = jit.lookup("matmul")
matmul(args)

print(c.numpy())
